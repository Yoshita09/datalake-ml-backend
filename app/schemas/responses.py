<<<<<<< HEAD
from typing import Literal, Optional
=======
>>>>>>> parent of ff6206f (configured model service)
from pydantic import BaseModel
from typing import Optional

class HeadMovementResponse(BaseModel):
    success: bool
    message: Optional[str] = None


class BlinkDetectionResponse(BaseModel):
    success: bool
    blink_count: Optional[int] = None
    message: Optional[str] = None

class FaceRecognitionResponse(BaseModel):
<<<<<<< HEAD
    success: bool
    message: str
    identity: Optional[str] = None
    score: Optional[float] = None
=======
    matched: bool
    confidence: float
    user_id: Optional[str] = None
    message: Optional[str] = None
>>>>>>> parent of ff6206f (configured model service)
