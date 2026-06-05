"""
backend/app/services/head_movement_service.py

Per-session head-movement challenge-response state machine.

Stage sequence:
    look_straight → turn_left → center → turn_right → final_center → verified

Each call to detect_head_movement() advances the state for the given session
by exactly one step when the correct head direction is detected.

success=True is returned only at the "verified" transition, after the liveness
model passes.  All intermediate transitions return success=False with the next
expected stage so the frontend can update its UI card.

Session state is stored in a plain dict keyed by session_id.  For production
with multiple workers, replace _session_states with Redis or a shared cache.
"""

import threading
import cv2
import torch
import torch.nn as nn

from torchvision import models, transforms


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ─────────────────────────────────────────────────────────────────────────────
# Liveness Model
# ─────────────────────────────────────────────────────────────────────────────

liveness_model = models.mobilenet_v3_small(
    weights=None
)

num_features = (
    liveness_model.classifier[3].in_features
)

liveness_model.classifier[3] = nn.Linear(
    num_features,
    2,
)


# Load Weights

MODEL_PATH = "models/head_movement.pth"

try:

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    liveness_model.load_state_dict(
        checkpoint
    )

    liveness_model = (
        liveness_model
        .to(device)
        .eval()
    )

    print("✅ Liveness model loaded")

except Exception as e:

    print("❌ Model load failed:", e)

    liveness_model = None


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.ToPILImage(),

    transforms.Resize((224, 224)),

    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# Haar Cascades
# ─────────────────────────────────────────────────────────────────────────────

front_face   = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
profile_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")


# ─────────────────────────────────────────────────────────────────────────────
# Per-session State Store
#
# Key  : session_id (UUID string from the mobile client)
# Value: { "stage": str, "completed": bool }
#
# Thread-safe via a lock; replace with Redis for multi-worker deployments.
# ─────────────────────────────────────────────────────────────────────────────

_state_lock    = threading.Lock()
_session_states: dict[str, dict] = {}

# Stage order — used for validation only; transitions are handled explicitly.
STAGE_ORDER = [
    "look_straight",
    "turn_left",
    "center",
    "turn_right",
    "final_center",
    "verified",
]


def _get_state(session_id: str) -> dict:
    """Return (and lazily create) the state dict for a session."""
    with _state_lock:
        if session_id not in _session_states:
            _session_states[session_id] = {
                "stage":     "look_straight",
                "completed": False,
            }
        return _session_states[session_id]


def _set_stage(session_id: str, stage: str) -> None:
    with _state_lock:
        if session_id in _session_states:
            _session_states[session_id]["stage"] = stage


def reset_session(session_id: str) -> None:
    """Reset a session to the initial state (call on client reset / retry)."""
    with _state_lock:
        _session_states[session_id] = {
            "stage":     "look_straight",
            "completed": False,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Direction Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_direction(gray) -> str:
    """
    Returns "center" | "left" | "right" | "unknown".

    Uses:
      - frontal cascade  → "center"
      - profile cascade  → "left"  (direct)
      - profile cascade  → "right" (on horizontally-flipped frame)
    """
    front_faces = front_face.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(80, 80),
    )
    if len(front_faces) > 0:
        return "center"

    left_profiles = profile_face.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(80, 80),
    )
    if len(left_profiles) > 0:
        return "left"

    flipped = cv2.flip(gray, 1)
    right_profiles = profile_face.detectMultiScale(
        flipped, scaleFactor=1.1, minNeighbors=3, minSize=(80, 80),
    )
    if len(right_profiles) > 0:
        return "right"
    
    print("LEFT:", len(left_profiles))
    print("RIGHT:", len(right_profiles))
    print("FRONT:", len(front_faces))

    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Liveness Check
# ─────────────────────────────────────────────────────────────────────────────

def run_liveness(frame) -> tuple[bool, float]:
    """
    Runs the MobileNetV3-Small liveness classifier on a BGR frame.

    Returns (is_live: bool, confidence: float 0–100).

    Model convention (adjust if yours differs):
        class 1 = LIVE
        class 0 = SPOOF
    """
    rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs       = liveness_model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        _, prediction = torch.max(outputs, 1)

    predicted_class = prediction.item()
    confidence      = probabilities[predicted_class].item() * 100

    print(f"Liveness — class: {predicted_class}, confidence: {confidence:.1f}%")

    # class 1 = LIVE
    return predicted_class == 1, confidence


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def detect_head_movement(frame, session_id: str) -> dict:
    """
    Advance the per-session head-movement state machine by one step.

    Args:
        frame      : BGR numpy array decoded from the incoming base64 JPEG.
        session_id : UUID string sent by the mobile client in every request.

    Returns a dict matching HeadMovementResponse:
        {
            "success":    bool,
            "stage":      str,   # next expected stage
            "message":    str,
            "confidence": float | None,
        }

    success=True is only returned when stage reaches "verified" and the
    liveness model confirms a live face.  All other transitions return
    success=False with the next expected stage for the frontend to display.
    """

    if liveness_model is None:

        return {
            "success": False,
            "message": "Model not loaded",
        }

    try:
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        direction = detect_direction(gray)
        state     = _get_state(session_id)
        stage     = state["stage"]

        print(f"[{session_id[:8]}] stage={stage} direction={direction}")

        # ── look_straight ──────────────────────────────────────────────────
        if stage == "look_straight":
            if direction == "center":
                _set_stage(session_id, "turn_left")
                return {
                    "success": False,
                    "stage":   "turn_left",
                    "message": "Turn your head LEFT",
                }

        # ── turn_left ──────────────────────────────────────────────────────
        elif stage == "turn_left":
            if direction == "left":
                _set_stage(session_id, "center")
                return {
                    "success": False,
                    "stage":   "center",
                    "message": "Return to center",
                }

        # ── center ─────────────────────────────────────────────────────────
        elif stage == "center":
            if direction == "center":
                _set_stage(session_id, "turn_right")
                return {
                    "success": False,
                    "stage":   "turn_right",
                    "message": "Turn your head RIGHT",
                }

        # ── turn_right ─────────────────────────────────────────────────────
        elif stage == "turn_right":
            if direction == "right":
                _set_stage(session_id, "final_center")
                return {
                    "success": False,
                    "stage":   "final_center",
                    "message": "Look straight again",
                }

        # ── final_center — run liveness ────────────────────────────────────
        elif stage == "final_center":
            if direction == "center":
                live, confidence = run_liveness(frame)

                if live:
                    with _state_lock:
                        _session_states[session_id]["stage"]     = "verified"
                        _session_states[session_id]["completed"] = True

                    return {
                        "success":    True,
                        "stage":      "verified",
                        "message":    "Liveness verified",
                        "confidence": confidence,
                    }
                else:
                    # Spoof detected — reset this session so the client can retry
                    reset_session(session_id)
                    return {
                        "success": False,
                        "stage":   "look_straight",
                        "message": "Spoof detected — please try again",
                    }

        # Fallback: direction not yet matched for current stage
        return {
            "success": False,
            "stage":   stage,
            "message": "Waiting for correct movement",
        }

    except Exception as e:

        print("Inference error:", e)

        return {
            "success": False,
            "message": str(e),
        }