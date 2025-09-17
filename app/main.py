# app/main.py
import os
import re, uuid, pathlib, time

# --- CPU only / env flags (não inicializa CUDA/TF) ---
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("INSIGHTFACE_HOME", "/workspace/.insightface")
os.environ.setdefault("FORCE_OPENCV_DETECTOR", "1")  # força OpenCV por padrão
os.environ.setdefault("RETURN_200_ON_ERRORS", "1")   # evita 500 em register/verify por padrão

import logging
import traceback
from contextvars import ContextVar
from secrets import token_hex
from typing import Optional, List, Dict

import cv2  # noqa: F401 - garante OpenCV carregado
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form, Request
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    LivenessResponse, LivenessQuery, FaceResult, BBox,
    RegisterResponse, VerifyResponse,
    DatasetUploadResponse, DatasetStatsResponse
)
from .utils import read_image_from_upload, read_image_from_base64
from .liveness import check_liveness, DEFAULT_DETECTOR, DEFAULT_THRESHOLD
from .recognition import (
    embed_face, save_embedding, load_embedding,
    cosine_sim, init_recognition
)

# ----------------- logging estruturado -----------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s [req=%(request_id)s] %(message)s",
)
_request_id: ContextVar[str] = ContextVar("request_id", default="-")

_old_factory = logging.getLogRecordFactory()
def _record_factory(*args, **kwargs):
    record = _old_factory(*args, **kwargs)
    record.request_id = _request_id.get("-")
    return record
logging.setLogRecordFactory(_record_factory)

log = logging.getLogger("api")

APP_NAME = "anti-spoofing-api"
VERSION = os.getenv("APP_VERSION", "1.0.0")

app = FastAPI(title=APP_NAME, version=VERSION)

# --------------- CORS ---------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --------------- middleware de request/response ---------------
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    rid = request.headers.get("x-request-id") or token_hex(8)
    _request_id.set(rid)
    start = time.time()
    client_ip = request.client.host if request.client else "-"
    try:
        log.info(f"--> {request.method} {request.url.path} from={client_ip}")
        resp = await call_next(request)
        dur = int((time.time() - start) * 1000)
        log.info(f"<-- {request.method} {request.url.path} status={resp.status_code} dur_ms={dur}")
        return resp
    except Exception as e:
        dur = int((time.time() - start) * 1000)
        log.exception(f"!! {request.method} {request.url.path} crashed after {dur} ms: {e}")
        raise

# --------------- utils ---------------
def _cuda_info():
    # não inicializa torch/tf aqui
    return {"available": False, "version": None, "device_count": 0}

def _area(b: Dict[str, int]) -> int:
    return int(b.get("w", 0)) * int(b.get("h", 0))

def _pick_primary_face(faces: List[Dict]) -> Optional[Dict]:
    if not faces:
        return None
    return max(faces, key=lambda r: _area(r["bbox"]))

def _crop_by_bbox(bgr: np.ndarray, bbox: Dict[str, int], margin: float = 0.2) -> np.ndarray:
    H, W = bgr.shape[:2]
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    cx, cy = x + w / 2.0, y + h / 2.0
    s = int(max(w, h) * (1.0 + margin))
    x1 = max(0, int(cx - s / 2)); y1 = max(0, int(cy - s / 2))
    x2 = min(W, int(cx + s / 2)); y2 = min(H, int(cy + s / 2))
    if x2 <= x1 or y2 <= y1:
        return bgr
    return bgr[y1:y2, x1:x2].copy()

def _resolve_params(
    detector_backend_q: Optional[str],
    threshold_q: Optional[float],
    detector_backend_f: Optional[str],
    threshold_f: Optional[float],
    body: Optional[LivenessQuery],
):
    detector = (body.detector_backend if body else None) or detector_backend_q or detector_backend_f or DEFAULT_DETECTOR
    thr = (body.threshold if body else None) or threshold_q or threshold_f or DEFAULT_THRESHOLD
    if os.getenv("FORCE_OPENCV_DETECTOR", "1") == "1":
        detector = "opencv"
    return detector, float(thr)

# --------------- startup (opcional) ---------------
@app.on_event("startup")
def _warmup_models():
    try:
        if os.getenv("DISABLE_WARMUP", "0") == "1":
            log.info("[warmup] disabled")
            return
        log.info("[warmup] begin")
        init_recognition(providers=["CPUExecutionProvider"])  # idempotente
        log.info("[warmup] done")
    except Exception as e:
        log.warning(f"[warmup] aviso: {e}")

# --------------- Health ---------------
@app.get("/health")
def health():
    import platform, sys
    flags = {
        "FORCE_OPENCV_DETECTOR": os.getenv("FORCE_OPENCV_DETECTOR"),
        "RETURN_200_ON_ERRORS": os.getenv("RETURN_200_ON_ERRORS"),
        "DISABLE_WARMUP": os.getenv("DISABLE_WARMUP", "0"),
        "LOG_LEVEL": LOG_LEVEL,
    }
    return {
        "status": "ok",
        "name": APP_NAME,
        "version": VERSION,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda": _cuda_info(),
        "flags": flags,
    }

# --------------- Liveness ---------------
@app.post("/v1/liveness", response_model=LivenessResponse)
async def v1_liveness(
    image: Optional[UploadFile] = File(default=None),
    # query
    detector_backend_q: Optional[str] = Query(None),
    threshold_q: Optional[float] = Query(None),
    # form
    detector_backend_f: Optional[str] = Form(None),
    threshold_f: Optional[float] = Form(None),
    # json
    body: Optional[LivenessQuery] = None,
):
    try:
        if image and body:
            raise HTTPException(status_code=400, detail="Envie apenas multipart OU JSON, não ambos.")
        if (image is None) and (not body):
            raise HTTPException(status_code=400, detail="Envie 'image' (multipart) OU 'image_base64' no JSON.")

        bgr = read_image_from_upload(image.file) if image else read_image_from_base64(body.image_base64)
        detector_backend, threshold = _resolve_params(detector_backend_q, threshold_q, detector_backend_f, threshold_f, body)

        log.info(f"[liveness] backend={detector_backend} thr={threshold}")
        results, latency_ms = check_liveness(bgr_image=bgr, detector_backend=detector_backend, threshold=threshold)
        log.info(f"[liveness] faces={len(results)} latency_ms={latency_ms}")
        if results:
            f0 = results[0]; bb = f0['bbox']
            log.info(f"[liveness] f0 is_live={f0['is_live']} score={f0['score']:.3f} bbox={bb['w']}x{bb['h']}@{bb['x']},{bb['y']}")

        faces = [FaceResult(
            is_live=r["is_live"],
            score=r["score"],
            threshold=r["threshold"],
            bbox=BBox(**r["bbox"]),
            extra={**(r.get("extra") or {}), "detector_used": detector_backend},
        ) for r in results]

        return LivenessResponse(faces=faces, latency_ms=latency_ms)

    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"[liveness] error: {e}")
        raise HTTPException(status_code=500, detail=f"Erro no processamento: {e}")

# --------------- Register ---------------
@app.post("/v1/register", response_model=RegisterResponse)
async def v1_register(
    image: Optional[UploadFile] = File(default=None),
    user_id: str = Query(..., description="ID único do usuário"),
    # query
    detector_backend_q: Optional[str] = Query(None),
    threshold_q: Optional[float] = Query(None),
    # form
    detector_backend_f: Optional[str] = Form(None),
    threshold_f: Optional[float] = Form(None),
    # json
    body: Optional[LivenessQuery] = None,
):
    try:
        if image and body:
            raise HTTPException(status_code=400, detail="Envie apenas multipart OU JSON.")
        if (image is None) and (not body):
            raise HTTPException(status_code=400, detail="Envie 'image' ou 'image_base64'.")

        bgr = read_image_from_upload(image.file) if image else read_image_from_base64(body.image_base64)
        detector_backend, threshold = _resolve_params(detector_backend_q, threshold_q, detector_backend_f, threshold_f, body)
        log.info(f"[register] user={user_id} backend={detector_backend} thr={threshold}")

        # 1) Liveness
        faces, _ = check_liveness(bgr, detector_backend=detector_backend, threshold=threshold)
        log.info(f"[register] faces={len(faces)}")
        primary = _pick_primary_face(faces)
        if not primary:
            msg = "Nenhum rosto válido"
            log.warning(f"[register] {msg}")
            return RegisterResponse(success=False, user_id=user_id, liveness_ok=False, message=msg)
        if not primary["is_live"]:
            msg = "Anti-spoofing reprovado"
            log.warning(f"[register] {msg}")
            return RegisterResponse(success=False, user_id=user_id, liveness_ok=False, message=msg)

        # 2) Embedding
        crop = _crop_by_bbox(bgr, primary["bbox"], margin=0.2)
        init_recognition(providers=["CPUExecutionProvider"])
        emb = embed_face(crop)

        save_embedding(user_id, emb)
        log.info(f"[register] saved embedding for user={user_id}")
        return RegisterResponse(success=True, user_id=user_id, liveness_ok=True, message="Cadastrado com sucesso")

    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"[register] error: {e}")
        if os.getenv("RETURN_200_ON_ERRORS", "1") == "1":
            return RegisterResponse(success=False, user_id=user_id, liveness_ok=False, message=f"Falha interna no cadastro: {e}")
        raise HTTPException(status_code=500, detail=f"Erro no cadastro: {e}")

# --------------- Verify ---------------
@app.post("/v1/verify", response_model=VerifyResponse)
async def v1_verify(
    image: Optional[UploadFile] = File(default=None),
    user_id: str = Query(..., description="ID do usuário a verificar"),
    # query
    detector_backend_q: Optional[str] = Query(None),
    threshold_q: Optional[float] = Query(None),
    match_threshold_q: Optional[float] = Query(0.35),
    # form
    detector_backend_f: Optional[str] = Form(None),
    threshold_f: Optional[float] = Form(None),
    match_threshold_f: Optional[float] = Form(None),
    # json
    body: Optional[LivenessQuery] = None,
):
    try:
        if image and body:
            raise HTTPException(status_code=400, detail="Envie apenas multipart OU JSON.")
        if (image is None) and (not body):
            raise HTTPException(status_code=400, detail="Envie 'image' ou 'image_base64'.")

        bgr = read_image_from_upload(image.file) if image else read_image_from_base64(body.image_base64)
        detector_backend, threshold = _resolve_params(detector_backend_q, threshold_q, detector_backend_f, threshold_f, body)
        match_threshold = float(match_threshold_q if match_threshold_q is not None else (match_threshold_f if match_threshold_f is not None else 0.35))
        log.info(f"[verify] user={user_id} backend={detector_backend} thr={threshold} match_thr={match_threshold}")

        # 1) Liveness
        faces, _ = check_liveness(bgr, detector_backend=detector_backend, threshold=threshold)
        log.info(f"[verify] faces={len(faces)}")
        primary = _pick_primary_face(faces)
        if not primary:
            msg = "Nenhum rosto válido"
            log.warning(f"[verify] {msg}")
            return VerifyResponse(success=False, user_id=user_id, liveness_ok=False, match=False,
                                  cosine_similarity=0.0, cosine_distance=1.0,
                                  match_threshold=match_threshold, message=msg)
        if not primary["is_live"]:
            msg = "Anti-spoofing reprovado"
            log.warning(f"[verify] {msg}")
            return VerifyResponse(success=False, user_id=user_id, liveness_ok=False, match=False,
                                  cosine_similarity=0.0, cosine_distance=1.0,
                                  match_threshold=match_threshold, message=msg)

        # 2) Embedding atual
        crop = _crop_by_bbox(bgr, primary["bbox"], margin=0.2)
        init_recognition(providers=["CPUExecutionProvider"])
        emb_now = embed_face(crop)

        # 3) Embedding cadastrado
        try:
            emb_ref = load_embedding(user_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Usuário não cadastrado")

        # 4) Métricas
        sim = cosine_sim(emb_now, emb_ref)
        dist = 1.0 - sim
        match = bool(dist <= match_threshold)

        log.info(f"[verify] user={user_id} match={match} cos_sim={sim:.6f} cos_dist={dist:.6f} thr={match_threshold}")
        return VerifyResponse(
            success=True, user_id=user_id, liveness_ok=True, match=match,
            cosine_similarity=sim, cosine_distance=dist, match_threshold=match_threshold,
            message="OK" if match else "Não corresponde"
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"[verify] error: {e}")
        if os.getenv("RETURN_200_ON_ERRORS", "1") == "1":
            return VerifyResponse(success=False, user_id=user_id, liveness_ok=False, match=False,
                                  cosine_similarity=0.0, cosine_distance=1.0,
                                  match_threshold=float(match_threshold_q or match_threshold_f or 0.35),
                                  message=f"Falha interna na verificação: {e}")
        raise HTTPException(status_code=500, detail=f"Erro na verificação: {e}")

# ---------- Dataset: upload e stats ----------
def _dataset_base() -> str:
    # raiz do dataset (env configurável)
    return os.getenv("TRAIN_DIR", "/workspace/data/train")

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", (s or "anon"))[:60] or "anon"

def _ext_from_filename(name: str) -> str:
    ext = (os.path.splitext(name or "")[1] or "").lower()
    return ext if ext in [".jpg", ".jpeg", ".png"] else ".jpg"

def _imwrite(path: str, bgr: np.ndarray) -> None:
    ext = os.path.splitext(path)[1].lower()
    flag = ".png" if ext == ".png" else ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), 92] if flag == ".jpg" else []
    ok, buf = cv2.imencode(flag, bgr, params)
    if not ok:
        raise RuntimeError("Falha ao codificar imagem")
    with open(path, "wb") as f:
        f.write(buf.tobytes())

@app.post("/v1/dataset/upload", response_model=DatasetUploadResponse)
async def dataset_upload(
    image: UploadFile = File(...),
    label: str = Query(..., description="live ou spoof"),
    user_id: Optional[str] = Query(None, description="opcional, referência"),
    # aceita parâmetros de liveness iguais aos outros endpoints
    detector_backend_q: Optional[str] = Query(None),
    threshold_q: Optional[float] = Query(None),
    detector_backend_f: Optional[str] = Form(None),
    threshold_f: Optional[float] = Form(None),
):
    """
    Salva a imagem em data/train/<label>/{raw,faces}, roda a detecção/liveness
    para validar e também salvar o crop da face.
    """
    label = (label or "").strip().lower()
    if label not in ("live", "spoof"):
        raise HTTPException(status_code=400, detail="label deve ser 'live' ou 'spoof'")

    try:
        bgr = read_image_from_upload(image.file)
        # usa a mesma resolução de parâmetros do /v1/liveness
        detector_backend, threshold = _resolve_params(
            detector_backend_q, threshold_q, detector_backend_f, threshold_f, body=None
        )
        results, _lat = check_liveness(bgr_image=bgr, detector_backend=detector_backend, threshold=threshold)
        if not results:
            raise HTTPException(status_code=422, detail="Nenhum rosto detectado")

        # pega a maior face
        primary = max(results, key=lambda r: r["bbox"]["w"] * r["bbox"]["h"])
        bbox = primary["bbox"]

        # paths
        base = _dataset_base()
        raw_dir   = _ensure_dir(os.path.join(base, label, "raw"))
        faces_dir = _ensure_dir(os.path.join(base, label, "faces"))

        ts  = int(time.time())
        uid = _safe_name(user_id or "anon")
        rnd = uuid.uuid4().hex[:8]
        ext = _ext_from_filename(image.filename)

        raw_path  = os.path.join(raw_dir,   f"{ts}_{uid}_{rnd}{ext}")
        face_path = os.path.join(faces_dir, f"{ts}_{uid}_{rnd}{ext}")

        # salva RAW
        _imwrite(raw_path, bgr)

        # recorta face com margem e salva
        crop = _crop_by_bbox(bgr, bbox, margin=0.2)
        _imwrite(face_path, crop)

        face_resp = FaceResult(
            is_live=primary["is_live"],
            score=float(primary["score"]),
            threshold=float(primary["threshold"]),
            bbox=BBox(**bbox),
            extra={**(primary.get("extra") or {}), "detector_used": detector_backend},
        )

        return DatasetUploadResponse(
            success=True,
            label=label,
            paths={
                "raw": str(pathlib.Path(raw_path).resolve()),
                "face": str(pathlib.Path(face_path).resolve()),
            },
            face=face_resp,
            message="Imagem adicionada ao dataset"
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"[dataset-upload] error: {e}")
        raise HTTPException(status_code=500, detail=f"Falha no upload do dataset: {e}")

@app.get("/v1/dataset/stats", response_model=DatasetStatsResponse)
def dataset_stats():
    base = pathlib.Path(_dataset_base())
    live_glob  = list((base / "live").rglob("*.[jp][pn]g"))
    spoof_glob = list((base / "spoof").rglob("*.[jp][pn]g"))

    n_live  = len(live_glob)
    n_spoof = len(spoof_glob)
    total   = n_live + n_spoof

    live_samples  = [str(p.resolve()) for p in sorted(live_glob)[-5:]]
    spoof_samples = [str(p.resolve()) for p in sorted(spoof_glob)[-5:]]

    return DatasetStatsResponse(
        total=total,
        live=n_live,
        spoof=n_spoof,
        samples={"live": live_samples, "spoof": spoof_samples},
    )
