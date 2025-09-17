# app/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

# ---------- Liveness ----------
class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int

class FaceResult(BaseModel):
    is_live: bool = Field(..., description="True = rosto real; False = spoof")
    score: float = Field(..., description="Pontuação de vivacidade (ou 1.0/0.0 quando a lib só retorna boolean)")
    threshold: float = Field(..., description="Limiar usado para decisão de vivacidade")
    bbox: BBox
    extra: Optional[Dict] = None

class LivenessResponse(BaseModel):
    faces: List[FaceResult]
    latency_ms: int

class LivenessQuery(BaseModel):
    image_base64: str
    detector_backend: Optional[str] = "retinaface"
    threshold: Optional[float] = 0.5

# ---------- Register / Verify ----------
class RegisterResponse(BaseModel):
    success: bool
    user_id: str
    liveness_ok: bool
    message: Optional[str] = None

class VerifyResponse(BaseModel):
    success: bool
    user_id: str
    liveness_ok: bool
    match: bool
    cosine_similarity: float
    cosine_distance: float
    match_threshold: float
    message: Optional[str] = None
