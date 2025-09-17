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

# -------- Detector OpenCV DNN (cache) --------
PROTOTXT = "deploy.prototxt"
WEIGHTS  = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
PROTOTXT_URLS = [
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy_lowres.prototxt",
]
CAFFEMODEL_URLS = [
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_models_201907/opencv_face_detector.caffemodel",
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel",
]
_DNN_NET = None
_DNN_READY = False

def _model_dir() -> str:
    root = os.environ.get("MODEL_DIR", "/workspace/models")
    path = os.path.join(root, "opencv_face")
    os.makedirs(path, exist_ok=True)
    return path

def _download(url: str, dst: str) -> None:
    import urllib.request
    log.info(f"[liveness-dnn] download {url} -> {dst}")
    urllib.request.urlretrieve(url, dst)

def _get_dnn_paths() -> Tuple[str, str]:
    p = _model_dir()
    proto = os.path.join(p, PROTOTXT)
    weights = os.path.join(p, WEIGHTS)
    if not os.path.exists(proto):
        ok = False
        for u in PROTOTXT_URLS:
            try: _download(u, proto); ok = True; break
            except Exception as e: log.warning(f"[dnn] prototxt fail {u}: {e}")
        if not ok: raise RuntimeError("Falha ao obter prototxt do detector.")
    if not os.path.exists(weights):
        ok = False
        for u in CAFFEMODEL_URLS:
            try: _download(u, weights); ok = True; break
            except Exception as e: log.warning(f"[dnn] caffemodel fail {u}: {e}")
        if not ok: raise RuntimeError("Falha ao obter caffemodel do detector.")
    return proto, weights

def _ensure_dnn_loaded():
    global _DNN_NET, _DNN_READY
    if _DNN_READY and _DNN_NET is not None:
        return
    proto, weights = _get_dnn_paths()
    _DNN_NET = cv2.dnn.readNetFromCaffe(proto, weights)
    _DNN_READY = True

def _detect_faces_dnn(bgr: np.ndarray, conf_thr: float = 0.5) -> List[Dict]:
    _ensure_dnn_loaded()
    H, W = bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(bgr, (300, 300)),
                                 1.0, (300, 300),
                                 mean=(104.0,177.0,123.0), swapRB=False, crop=False)
    _DNN_NET.setInput(blob)
    detections = _DNN_NET.forward()
    out = []
    if detections.ndim == 4:
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < conf_thr: continue
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H], dtype=np.float32)
            x1, y1, x2, y2 = box.astype(int)
            x, y = max(0, x1), max(0, y1)
            w, h = max(0, x2 - x1), max(0, y2 - y1)
            if w >= 30 and h >= 30:
                out.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
    return out

def _detect_faces_haar(bgr: np.ndarray) -> List[Dict]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                     minSize=(80,80), flags=cv2.CASCADE_SCALE_IMAGE)
    return [{"x":int(x), "y":int(y), "w":int(w), "h":int(h)} for (x,y,w,h) in faces]

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
    try:
        faces = _detect_faces_dnn(img, conf_thr=0.5)
        log.info(f"[liveness-core] faces_dnn={len(faces)}")
        if not faces:
            faces = _detect_faces_haar(img)
            log.info(f"[liveness-core] faces_haar={len(faces)}")
    except Exception as e:
        log.warning(f"[liveness-core] DNN erro: {e}; fallback Haar")
        try:
            faces = _detect_faces_haar(img)
            log.info(f"[liveness-core] faces_haar={len(faces)}")
        except Exception as e2:
            latency = _now_ms() - t0
            log.exception(f"[liveness-core] sem detecção após {latency} ms: {e2}")
            return [], latency

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
            "extra": {"detector_backend": "opencv-dnn", "explain": detail}
        })
        log.info(f"[liveness-core] face#{i} score={sc:.3f} live={live} thr_eff={dyn_thr:.3f} "
                 f"axis={detail['axis_anisotropy']:.2f} glare={detail['glare']:.2f} lines={detail['line_ratio']:.2f}")

    latency = _now_ms() - t0
    log.info(f"[liveness-core] done faces={len(results)} latency_ms={latency}")
    return results, latency
