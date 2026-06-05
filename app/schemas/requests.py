from pydantic import BaseModel


class FrameRequest(BaseModel):
    frame: str
    timestamp: str
    session_id: str
    step: str