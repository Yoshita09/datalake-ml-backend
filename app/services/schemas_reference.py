"""
backend/app/schemas/  — reference snippets

These are the Pydantic models the updated router and service expect.
Add/merge them into your existing schemas files.
"""

from typing import Literal, Optional
from pydantic import BaseModel


# ── requests.py ──────────────────────────────────────────────────────────────

class FrameRequest(BaseModel):
    """Sent by the mobile client for every ML step."""
    frame:      str   # base64 JPEG, no data-URL prefix
    timestamp:  str   # ISO-8601 string
    session_id: str   # UUID — ties frames to a session on the backend
    step:       str   # "head_movement" | "blink_detection" | "face_recognition"


# ── responses.py ─────────────────────────────────────────────────────────────

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

    success  = True only when stage == "verified" and liveness passes.
    stage    = the stage the backend has advanced to; the frontend uses this
               to update the HeadMovement card immediately.
    message  = human-readable status (for logging / debug toasts).
    confidence = liveness confidence 0–100, present only on verified response.
    """
    success:    bool
    stage:      Optional[HeadStage] = None
    message:    Optional[str]       = None
    confidence: Optional[float]     = None