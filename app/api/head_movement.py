"""
backend/app/api/head_movement.py

FastAPI router for the /head-movement endpoint.

The session_id from FrameRequest is forwarded to the service layer so each
mobile client gets its own isolated state machine — no cross-session bleed.
"""

from fastapi import APIRouter

from app.schemas.requests  import FrameRequest
from app.schemas.responses import HeadMovementResponse

from app.services.frame_decoder        import decode_base64_frame
from app.services.head_movement_service import detect_head_movement

router = APIRouter()


@router.post(
    "/head-movement",
    response_model=HeadMovementResponse,
)
async def head_movement(payload: FrameRequest):
    """
    Receives a base64 JPEG frame and the client's session_id.

    The service layer advances the per-session state machine by one step
    and returns the next expected stage so the mobile UI can update its
    HeadMovement card in real time.
    """
    frame = decode_base64_frame(payload.frame)

    result = detect_head_movement(frame, session_id=payload.session_id)

    return result