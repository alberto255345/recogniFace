# app/recognition.py
import os
from typing import Optional, Tuple, Dict
import numpy as np
from numpy.linalg import norm
import cv2

# >>> usamos DeepFace (Facenet512) para embeddings - CPU <<<
from deepface import DeepFace

# Stub para compatibilidade com main.py (não faz nada, só existe)
def init_recognition(providers=None, det_size: Tuple[int, int] = (640, 640)):
    return None

def _ensure_min_side(img: np.ndarray, min_side: int = 480) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    if s >= min_side:
        return img
    scale = float(min_side) / max(1, s)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def _crop_by_bbox(bgr: np.ndarray, bbox: Dict[str, int], margin: float = 0.35) -> np.ndarray:
    H, W = bgr.shape[:2]
    x = int(bbox.get("x", 0)); y = int(bbox.get("y", 0))
    w = int(bbox.get("w", 0)); h = int(bbox.get("h", 0))
    cx, cy = x + w / 2.0, y + h / 2.0
    s = int(max(w, h) * (1.0 + margin))
    x1 = max(0, int(cx - s / 2)); y1 = max(0, int(cy - s / 2))
    x2 = min(W, int(cx + s / 2)); y2 = min(H, int(cy + s / 2))
    if x2 <= x1 or y2 <= y1:
        return bgr
    return bgr[y1:y2, x1:x2].copy()

def _represent_bgr(bgr: np.ndarray) -> np.ndarray:
    """
    Extrai embedding com DeepFace (Facenet512).
    - detector_backend='skip' porque já recebemos a face recortada.
    - enforce_detection=False para não levantar exceção.
    """
    # DeepFace aceita np.ndarray em BGR; mantemos sem converter para RGB.
    reps = DeepFace.represent(
        img_path=bgr,
        model_name="Facenet512",
        detector_backend="skip",
        enforce_detection=False,
        align=False,
        normalization="base",
    )
    if not reps:
        raise ValueError("Falha ao extrair embedding (DeepFace vazio)")
    emb = np.array(reps[0]["embedding"], dtype="float32")
    # normaliza (cosine-friendly)
    emb = emb / (norm(emb) + 1e-9)
    return emb

def embed_face_from_bbox(bgr: np.ndarray, bbox: Dict[str, int], margin: float = 0.35) -> np.ndarray:
    """
    Tenta no crop (várias margens); se falhar, usa frame inteiro com upscale.
    """
    for m in (margin, 0.6, 1.0):
        try:
            crop = _crop_by_bbox(bgr, bbox, margin=m)
            crop = _ensure_min_side(crop, 560)
            return _represent_bgr(crop)
        except Exception:
            continue
    full = _ensure_min_side(bgr, 720)
    return _represent_bgr(full)

# Mantém nomes que o main importa (backwards compatible)
def embed_face(img_bgr: np.ndarray) -> np.ndarray:
    img = _ensure_min_side(img_bgr, 560)
    return _represent_bgr(img)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-9))

def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_sim(a, b)

def save_embedding(user_id: str, emb: np.ndarray, base_dir: str = "data/embeddings") -> str:
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"{user_id}.npz")
    np.savez_compressed(path, embedding=emb)
    return path

def load_embedding(user_id: str, base_dir: str = "data/embeddings") -> np.ndarray:
    path = os.path.join(base_dir, f"{user_id}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError("Usuário não cadastrado")
    data = np.load(path)
    return data["embedding"].astype("float32")
