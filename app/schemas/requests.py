# backend/app/schemas/requests.py

from pydantic import BaseModel


class FrameRequest(BaseModel):
    """Sent by the mobile client for every ML step."""
    frame:      str   # base64 JPEG, no data-URL prefix
    timestamp:  str   # ISO-8601 string
    session_id: str   # UUID — ties frames to a session on the backend
    step:       str   # "head_movement" | "blink_detection" | "face_recognition"
