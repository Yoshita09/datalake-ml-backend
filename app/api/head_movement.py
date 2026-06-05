from fastapi import APIRouter

from app.schemas.requests import FrameRequest
from app.schemas.responses import HeadMovementResponse

from app.services.frame_decoder import decode_base64_frame
from app.services.head_movement_service import detect_head_movement

router = APIRouter()


@router.post(
    "/head-movement",
    response_model=HeadMovementResponse,
)
async def head_movement(
    payload: FrameRequest,
):
    frame = decode_base64_frame(payload.frame)

    result = detect_head_movement(frame)

    return result