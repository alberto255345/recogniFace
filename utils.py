# app/utils.py  # NEW
import base64
import io
from PIL import Image
import numpy as np

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
