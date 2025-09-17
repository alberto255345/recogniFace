# app/utils.py  # NEW
import base64
import io
import os
import pathlib
import tempfile
from typing import List

import cv2
import numpy as np
from PIL import Image

def read_image_from_upload(file) -> np.ndarray:
    data = file.read()
    return _decode_bytes_to_bgr(data)

def read_image_from_base64(b64: str) -> np.ndarray:
    data = base64.b64decode(b64.split(",")[-1])
    return _decode_bytes_to_bgr(data)

def _decode_bytes_to_bgr(data: bytes) -> np.ndarray:
    # PIL -> RGB -> BGR (DeepFace aceita np.ndarray; OpenCV padrão é BGR)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.array(img)
    return arr[:, :, ::-1]  # RGB->BGR


def decode_video_to_frames(
    data: bytes,
    *,
    filename: str | None = None,
    max_frames: int = 24,
) -> List[np.ndarray]:
    """Decodifica um vídeo (bytes) em uma lista de frames BGR.

    Args:
        data: Conteúdo bruto do vídeo.
        filename: Nome original (usado para inferir extensão temporária).
        max_frames: Número máximo de frames retornados (<=0 = sem limite).

    Returns:
        Lista de frames (np.ndarray em BGR). Pode ser vazia se falhar.
    """
    if not data:
        return []

    suffix = pathlib.Path(filename).suffix if filename else ""
    if not suffix:
        suffix = ".mp4"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(tmp_path)
    try:
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target_frames = max_frames if max_frames > 0 else total_frames or 0
        if total_frames and target_frames:
            # espaçamento proporcional para cobrir o vídeo todo
            step = max(1, total_frames // target_frames)
        else:
            step = 1

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frames.append(frame)
                if max_frames > 0 and len(frames) >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return frames
