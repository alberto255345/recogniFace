# app/liveness.py
import os
import time
import math
import logging
from typing import List, Tuple, Dict

import cv2
import numpy as np

DEFAULT_DETECTOR = "opencv"
DEFAULT_THRESHOLD = 0.5

log = logging.getLogger("api")

# Estabilidade
try: cv2.setNumThreads(1)
except Exception: pass
try: cv2.ocl.setUseOpenCL(False)
except Exception: pass

# -------- Parâmetros ajustáveis por ENV --------
LIVENESS_MIN_SIDE = int(os.getenv("LIVENESS_MIN_SIDE", "224"))

# pesos “positivos” (aumentam score)
W_BLUR = float(os.getenv("LIVENESS_W_BLUR", "0.35"))
W_SAT  = float(os.getenv("LIVENESS_W_SAT",  "0.20"))
W_FREQ = float(os.getenv("LIVENESS_W_FREQ", "0.45"))

# pesos “negativos” (diminuem score)
W_AXIS  = float(os.getenv("LIVENESS_W_AXIS",  "0.25"))  # anisotropia 0/90°
W_GLARE = float(os.getenv("LIVENESS_W_GLARE", "0.12"))  # brilho alto + saturação baixa
W_LINES = float(os.getenv("LIVENESS_W_LINES", "0.15"))  # linhas retas longas
W_SHARP_SMALL = float(os.getenv("LIVENESS_W_SHARP_SMALL", "0.12"))  # super-nitidez em face pequena

# thresholds p/ elevar o limiar dinâmico quando suspeito
AXIS_SUSPECT = float(os.getenv("LIVENESS_AXIS_SUSPECT", "0.35"))
GLARE_SUSPECT = float(os.getenv("LIVENESS_GLARE_SUSPECT", "0.08"))
LINES_SUSPECT = float(os.getenv("LIVENESS_LINES_SUSPECT", "0.10"))

# -------- Utils --------
def _now_ms() -> int:
    return int(time.time() * 1000)

def _ensure_min_side(img: np.ndarray, min_side: int = LIVENESS_MIN_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    if s >= min_side: return img
    scale = float(min_side) / max(1, s)
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

def _pre_enhance(bgr: np.ndarray) -> np.ndarray:
    # CLAHE no Y + unsharp leve
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    y = yuv[..., 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y = clahe.apply(y)
    yuv[..., 0] = y
    bgr2 = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    g = cv2.GaussianBlur(bgr2, (0, 0), 1.0)
    return cv2.addWeighted(bgr2, 1.5, g, -0.5, 0)

def _mean_brightness(bgr: np.ndarray) -> float:
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)[..., 0]
    return float(np.mean(y))

_OPENCV_FACE_DIR = os.environ.get("OPENCV_FACE_DIR")
if not _OPENCV_FACE_DIR:
    model_dir = os.environ.get("MODEL_DIR")
    if model_dir:
        _OPENCV_FACE_DIR = os.path.join(model_dir, "opencv_face")
    else:
        _OPENCV_FACE_DIR = "models/opencv_face"
_OPENCV_FACE_DIR = os.path.expanduser(_OPENCV_FACE_DIR)
_OPENCV_DNN_DISABLE = os.environ.get("OPENCV_DNN_DISABLE", "0").lower() in {"1", "true", "yes"}
_OPENCV_DNN_CHECKED = False
_OPENCV_DNN_OK = False
_OPENCV_DNN_NET = None
_YUNET_CHECKED = False
_YUNET_AVAILABLE = False
_YUNET_DETECTOR = None
_last_log_ts = 0.0

YUNET_SCORE_THRESHOLD = float(os.environ.get("YUNET_SCORE_THRESHOLD", "0.6"))
YUNET_NMS_THRESHOLD = float(os.environ.get("YUNET_NMS_THRESHOLD", "0.3"))


def _log_throttle(prefix: str, msg: str, every_sec: int = 30) -> None:
    global _last_log_ts
    now = time.time()
    if now - _last_log_ts >= every_sec:
        log.info(f"{prefix} {msg}")
        _last_log_ts = now


def _dnn_files() -> Tuple[str, str]:
    prototxt = os.path.join(_OPENCV_FACE_DIR, "deploy.prototxt")
    caffemodel = os.path.join(_OPENCV_FACE_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
    return prototxt, caffemodel


def _check_dnn_available_once() -> bool:
    global _OPENCV_DNN_CHECKED, _OPENCV_DNN_OK
    if _OPENCV_DNN_CHECKED:
        return _OPENCV_DNN_OK
    if _OPENCV_DNN_DISABLE:
        _OPENCV_DNN_OK = False
    else:
        prototxt, caffemodel = _dnn_files()
        _OPENCV_DNN_OK = os.path.isfile(prototxt) and os.path.isfile(caffemodel)
        if not _OPENCV_DNN_OK:
            _log_throttle("[liveness-core]", "DNN Caffe indisponível localmente; usando Haar/YuNet.")
    _OPENCV_DNN_CHECKED = True
    return _OPENCV_DNN_OK


def _load_dnn_net():
    global _OPENCV_DNN_NET, _OPENCV_DNN_OK
    if _OPENCV_DNN_NET is not None:
        return _OPENCV_DNN_NET
    if not _check_dnn_available_once():
        return None
    prototxt, caffemodel = _dnn_files()
    try:
        _OPENCV_DNN_NET = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        return _OPENCV_DNN_NET
    except Exception as exc:
        log.warning(f"[liveness-dnn] erro ao carregar rede Caffe local: {exc}")
        _OPENCV_DNN_NET = None
        _OPENCV_DNN_OK = False
        return None


def _detect_faces_opencv_dnn(bgr: np.ndarray, conf_thr: float = 0.5) -> Tuple[List[Dict], str]:
    net = _load_dnn_net()
    if net is None:
        return [], "opencv-dnn-missing"

    H, W = bgr.shape[:2]
    try:
        blob = cv2.dnn.blobFromImage(bgr, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        net.setInput(blob)
        detections = net.forward()
    except Exception as exc:
        log.warning(f"[liveness-dnn] erro ao rodar DNN: {exc}")
        # evita novas tentativas neste processo
        global _OPENCV_DNN_CHECKED, _OPENCV_DNN_OK, _OPENCV_DNN_NET
        _OPENCV_DNN_NET = None
        _OPENCV_DNN_CHECKED = True
        _OPENCV_DNN_OK = False
        return [], "opencv-dnn-error"

    faces: List[Dict] = []
    if detections.ndim == 4:
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < conf_thr:
                continue
            x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * [W, H, W, H]).astype(int)
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            w, h = max(0, int(x2) - x1), max(0, int(y2) - y1)
            if w > 0 and h > 0:
                faces.append({"x": x1, "y": y1, "w": w, "h": h})
    return faces, "opencv-dnn"


def _detect_faces_haar(bgr: np.ndarray) -> Tuple[List[Dict], str]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    rects = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(64, 64),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    faces = [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in rects]
    return faces, "haar"


def _yunet_model_path() -> str:
    return os.path.join(_OPENCV_FACE_DIR, "face_detection_yunet_2023mar.onnx")


def _ensure_yunet_loaded():
    global _YUNET_CHECKED, _YUNET_AVAILABLE, _YUNET_DETECTOR
    if _YUNET_CHECKED and not _YUNET_AVAILABLE:
        return None
    if not hasattr(cv2, "FaceDetectorYN_create"):
        if not _YUNET_CHECKED:
            _log_throttle("[liveness-yunet]", "OpenCV não possui FaceDetectorYN_create; instale opencv-contrib-python.")
        _YUNET_CHECKED = True
        _YUNET_AVAILABLE = False
        _YUNET_DETECTOR = None
        return None
    model_path = _yunet_model_path()
    if not os.path.isfile(model_path):
        if not _YUNET_CHECKED:
            _log_throttle("[liveness-yunet]", f"Modelo YuNet não encontrado em {model_path}")
        _YUNET_CHECKED = True
        _YUNET_AVAILABLE = False
        _YUNET_DETECTOR = None
        return None
    if _YUNET_DETECTOR is None:
        try:
            _YUNET_DETECTOR = cv2.FaceDetectorYN_create(
                model=model_path,
                config="",
                input_size=(320, 320),
                score_threshold=YUNET_SCORE_THRESHOLD,
                nms_threshold=YUNET_NMS_THRESHOLD,
                top_k=500,
            )
            _YUNET_AVAILABLE = True
        except Exception as exc:
            _log_throttle("[liveness-yunet]", f"Falha ao carregar YuNet: {exc}")
            _YUNET_CHECKED = True
            _YUNET_AVAILABLE = False
            _YUNET_DETECTOR = None
            return None
    _YUNET_CHECKED = True
    return _YUNET_DETECTOR


def _detect_faces_yunet(bgr: np.ndarray) -> Tuple[List[Dict], str]:
    detector = _ensure_yunet_loaded()
    if detector is None:
        return [], "yunet-missing"
    h, w = bgr.shape[:2]
    try:
        detector.setInputSize((w, h))
        _, results = detector.detect(bgr)
    except Exception as exc:
        _log_throttle("[liveness-yunet]", f"Erro durante detecção YuNet: {exc}")
        return [], "yunet-error"
    faces: List[Dict] = []
    if results is not None:
        for row in results:
            x, y, ww, hh = map(int, row[:4])
            faces.append({"x": max(0, x), "y": max(0, y), "w": max(0, ww), "h": max(0, hh)})
    return faces, "yunet"


def detect_faces(bgr: np.ndarray, backend: str) -> Tuple[List[Dict], str]:
    backend = (backend or DEFAULT_DETECTOR).lower()
    if backend not in {"haar", "opencv", "yunet"}:
        backend = DEFAULT_DETECTOR
    if backend == "yunet":
        faces, used = _detect_faces_yunet(bgr)
        if used == "yunet":
            return faces, used
        faces, used = _detect_faces_opencv_dnn(bgr)
        if used == "opencv-dnn":
            return faces, used
        return _detect_faces_haar(bgr)
    if backend == "haar":
        return _detect_faces_haar(bgr)
    # padrão/opencv: tenta DNN e cai para Haar uma única vez
    faces, used = _detect_faces_opencv_dnn(bgr)
    if used == "opencv-dnn":
        return faces, used
    return _detect_faces_haar(bgr)

# -------- Métricas tradicionais --------
def _laplacian_blur_score(gray_roi: np.ndarray) -> float:
    val = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())
    return 1.0 / (1.0 + math.exp(-(val - 80.0)/20.0))  # 0..1 centrado ~80

def _saturation_score(bgr_roi: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    s = hsv[...,1].astype(np.float32) / 255.0
    return float(np.clip(np.mean(s), 0.0, 1.0))

def _freq_high_ratio(gray_roi: np.ndarray) -> float:
    g = gray_roi.astype(np.float32) / 255.0
    g = g - np.mean(g)
    if g.size == 0: return 0.0
    F = np.fft.fft2(g)
    Fshift = np.fft.fftshift(F)
    mag = np.abs(Fshift)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = min(cy, cx) // 4
    Y, X = np.ogrid[:h, :w]
    mask_lf = (X - cx)**2 + (Y - cy)**2 <= r**2
    sum_all = float(np.sum(mag) + 1e-6)
    sum_lf  = float(np.sum(mag[mask_lf]))
    sum_hf  = sum_all - sum_lf
    return float(np.clip(sum_hf / sum_all, 0.0, 1.0))

# -------- Novas métricas anti-spoof --------
def _axis_anisotropy(gray_roi: np.ndarray, theta_deg: float = 10.0) -> float:
    """Energia alta-freq concentrada nos eixos 0°/90° (padrão de grid)."""
    sz = 256
    g = cv2.resize(gray_roi, (sz, sz), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    g = g - np.mean(g)
    if np.allclose(g, 0): return 0.0
    F = np.fft.fft2(g)
    M = np.abs(np.fft.fftshift(F))
    h, w = M.shape
    cy, cx = h//2, w//2
    Y, X = np.ogrid[:h, :w]
    dy, dx = (Y - cy).astype(np.float32), (X - cx).astype(np.float32)
    ang = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 180.0  # 0..180
    R = np.sqrt(dx*dx + dy*dy)
    r0 = 16  # remove baixas freq
    high = (R > r0)

    def band(angle0):
        d = np.abs(ang - angle0)
        d = np.minimum(d, 180.0 - d)
        return (d <= theta_deg) & high

    axis_mask = band(0.0) | band(90.0)
    diag_mask = band(45.0) | band(135.0)
    E_total = float(np.sum(M[high]) + 1e-6)
    E_axis  = float(np.sum(M[axis_mask]))
    E_diag  = float(np.sum(M[diag_mask]))
    # anisotropia normalizada 0..1
    aniso = (E_axis - E_diag) / E_total
    return float(np.clip(0.5 + 0.5*aniso, 0.0, 1.0))

def _glare_score(bgr_roi: np.ndarray) -> float:
    """Pixéis muito claros com baixa saturação -> provável reflexo."""
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    mask = (V >= 240) & (S <= 35)  # 8-bit
    return float(np.clip(np.mean(mask.astype(np.float32)), 0.0, 1.0))

def _line_ratio(bgr_roi: np.ndarray) -> float:
    """Proporção de linhas retas longas (borda de papel/tela)."""
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(gray, 60, 180, L2gradient=True)
    h, w = gray.shape[:2]
    min_len = int(0.30 * min(h, w))
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=min_len, maxLineGap=10)
    if lines is None: return 0.0
    total_len = 0.0
    for l in lines[:,0,:]:
        x1,y1,x2,y2 = map(int, l)
        total_len += math.hypot(x2-x1, y2-y1)
    # normaliza pelo perímetro do ROI
    perim = 2.0*(h + w) + 1e-6
    return float(np.clip(total_len / perim, 0.0, 1.0))

def _sharp_small_penalty(blur_score: float, face_w: int, face_h: int) -> float:
    """Penaliza super-nitidez em face pequena."""
    m = min(face_w, face_h)
    if m >= 220 or blur_score <= 0.98:
        return 0.0
    # cresce conforme fica menor e mais “nítido”
    base = (blur_score - 0.98) / 0.02  # 0..1 quando 0.98..1.0
    size = np.clip((220 - m) / 80.0, 0.0, 1.0)
    return float(np.clip(base * size, 0.0, 1.0))

# -------- Score combinado --------
def _compute_liveness_score(bgr_roi: np.ndarray, face_w: int, face_h: int) -> Tuple[float, Dict]:
    roi = bgr_roi.copy()
    if min(roi.shape[:2]) < 80:
        scale = 80.0 / max(1, min(roi.shape[:2]))
        roi = cv2.resize(roi, (int(roi.shape[1]*scale), int(roi.shape[0]*scale)), interpolation=cv2.INTER_LINEAR)
    roi = _pre_enhance(roi)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    blur  = _laplacian_blur_score(gray)
    sat   = _saturation_score(roi)
    freq  = _freq_high_ratio(gray)

    # novas
    axis  = _axis_anisotropy(gray)   # maior -> mais parecido com tela
    glare = _glare_score(roi)        # fração de reflexo
    lines = _line_ratio(roi)         # proporção de linhas

    penalty_sharp = _sharp_small_penalty(blur, face_w, face_h)

    # score base
    score = W_BLUR*blur + W_SAT*sat + W_FREQ*freq
    # subtrai penalidades
    score -= (W_AXIS*axis + W_GLARE*glare + W_LINES*lines + W_SHARP_SMALL*penalty_sharp)
    score = float(np.clip(score, 0.0, 1.0))

    explain = {
        "blur": blur, "saturation": sat, "freq_high_ratio": freq,
        "axis_anisotropy": axis, "glare": glare, "line_ratio": lines,
        "penalty_sharp_small": penalty_sharp,
        "w": {
            "blur": W_BLUR, "saturation": W_SAT, "freq": W_FREQ,
            "axis": W_AXIS, "glare": W_GLARE, "lines": W_LINES, "sharp_small": W_SHARP_SMALL,
        }
    }
    return score, explain

# -------- Principal --------
def check_liveness(
    bgr_image: np.ndarray,
    detector_backend: str = DEFAULT_DETECTOR,
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[List[Dict], int]:
    t0 = _now_ms()

    if os.getenv("FORCE_OPENCV_DETECTOR", "1") == "1":
        detector_backend = "opencv"

    img = _ensure_min_side(bgr_image, LIVENESS_MIN_SIDE)
    H, W = img.shape[:2]
    log.info(f"[liveness-core] backend={detector_backend} img={W}x{H} thr={threshold}")

    results: List[Dict] = []
    # 1) detectar
    backend_norm = (detector_backend or DEFAULT_DETECTOR).lower()
    faces, detector_used = detect_faces(img, backend_norm)
    log.info(
        "[liveness-core] faces=%s detector_used=%s requested=%s",
        len(faces), detector_used, backend_norm,
    )

    # 2) métricas + limiar dinâmico (agora com aumento se “suspeito”)
    for i, bb in enumerate(faces):
        x, y, w, h = bb["x"], bb["y"], bb["w"], bb["h"]
        x2, y2 = min(W, x + w), min(H, y + h)
        roi = img[y:y2, x:x2].copy()
        roi = _ensure_min_side(roi, LIVENESS_MIN_SIDE)

        # limiar dinâmico básico (casos difíceis)
        dyn_thr = float(threshold)
        br = _mean_brightness(roi)
        if br < 90:       dyn_thr -= 0.10
        if max(w,h) < 260: dyn_thr -= 0.05

        # score e detalhes
        sc, detail = _compute_liveness_score(roi, w, h)

        # se sinais de spoof, sobe limiar (endurece a decisão)
        if detail["axis_anisotropy"] >= AXIS_SUSPECT: dyn_thr += 0.08
        if detail["glare"] >= GLARE_SUSPECT:          dyn_thr += 0.05
        if detail["line_ratio"] >= LINES_SUSPECT:     dyn_thr += 0.07
        dyn_thr = float(np.clip(dyn_thr, 0.10, 0.90))

        live = bool(sc >= dyn_thr)
        results.append({
            "is_live": live,
            "score": sc,
            "threshold": dyn_thr,
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "extra": {
                "detector_backend": detector_used,
                "detector_used": detector_used,
                "detector_requested": backend_norm,
                "explain": detail,
            }
        })
        log.info(f"[liveness-core] face#{i} score={sc:.3f} live={live} thr_eff={dyn_thr:.3f} "
                 f"axis={detail['axis_anisotropy']:.2f} glare={detail['glare']:.2f} lines={detail['line_ratio']:.2f}")

    latency = _now_ms() - t0
    log.info(f"[liveness-core] done faces={len(results)} latency_ms={latency}")
    return results, latency
