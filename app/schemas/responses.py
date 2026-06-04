# backend/app/schemas/responses.py

from typing import Literal, Optional
from pydantic import BaseModel


HeadStage = Literal[
    "look_straight",
    "turn_left",
    "center",
    "turn_right",
    "final_center",
    "verified",
]


class HeadMovementResponse(BaseModel):
    """
    Returned by POST /head-movement.

    success    = True only when stage == "verified" and liveness passes.
    stage      = stage the backend has advanced to; frontend updates its card.
    message    = human-readable status (for logging / debug toasts).
    confidence = liveness confidence 0–100, present only on verified response.
    """
    success:    bool
    stage:      Optional[HeadStage] = None
    message:    Optional[str]       = None
    confidence: Optional[float]     = None


class BlinkDetectionResponse(BaseModel):
    success:     bool
    blink_count: Optional[int] = None
    message:     Optional[str] = None


class FaceRecognitionResponse(BaseModel):
    matched:    bool
    confidence: float
    user_id:    Optional[str] = None
    message:    Optional[str] = None
