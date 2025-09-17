# app/recognition.py
from __future__ import annotations
import os
from typing import Optional, Tuple, List
import numpy as np
from numpy.linalg import norm
import cv2
from insightface.app import FaceAnalysis

_APP: Optional[FaceAnalysis] = None

def init_recognition(
    providers: Optional[List[str]] = None,
    det_size: Tuple[int, int] = (960, 960),
) -> FaceAnalysis:
    """
    Inicializa (uma única vez) o InsightFace (buffalo_l) via ONNXRuntime.
    Por padrão roda em CPU (providers=['CPUExecutionProvider']).
    """
    global _APP
    if _APP is not None:
        return _APP

    root = os.environ.get("INSIGHTFACE_HOME") or os.path.expanduser("~/.insightface")
    os.makedirs(root, exist_ok=True)

    if providers is None:
        providers = ["CPUExecutionProvider"]

    app = FaceAnalysis(name="buffalo_l", providers=providers, root=root)

    # ctx_id: -1 para CPU, 0+ para GPU. Se houver CUDAExecutionProvider, usa 0.
    ctx_id = 0 if any("CUDA" in p for p in providers) else -1
    app.prepare(ctx_id=ctx_id, det_size=det_size)

    _APP = app
    return _APP

def _ensure_app() -> FaceAnalysis:
    return init_recognition() if _APP is None else _APP

def _pick_largest(face_objs):
    def area(f):
        box = f.bbox.astype(int)
        return int((box[2] - box[0]) * (box[3] - box[1]))
    return max(face_objs, key=area)

def embed_face(bgr: np.ndarray) -> np.ndarray:
    """
    Recebe um recorte BGR contendo uma face e retorna embedding L2-normalizado (float32).
    """
    app = _ensure_app()
    if bgr is None or bgr.size == 0:
        raise ValueError("Imagem vazia para embedding")

    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype("uint8")

    faces = app.get(bgr)
    if not faces:
        raise ValueError("Nenhum rosto detectado para embedding")

    face = _pick_largest(faces)
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = face.embedding
        emb = emb / (norm(emb) + 1e-9)

    return emb.astype("float32")

def save_embedding(user_id: str, embedding: np.ndarray, base_dir: str = "data/embeddings") -> str:
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"{user_id}.npz")
    np.savez_compressed(path, embedding=embedding.astype("float32"))
    return path

def load_embedding(user_id: str, base_dir: str = "data/embeddings") -> np.ndarray:
    path = os.path.join(base_dir, f"{user_id}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path)
    emb = data["embedding"].astype("float32")
    emb = emb / (norm(emb) + 1e-9)
    return emb

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype("float32"); b = b.astype("float32")
    a = a / (norm(a) + 1e-9)
    b = b / (norm(b) + 1e-9)
    return float(np.dot(a, b))

def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_sim(a, b)
