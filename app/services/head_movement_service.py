"""
backend/app/services/head_movement_service.py

Per-session head-movement challenge-response state machine.

Stage sequence:
    look_straight → turn_left → center → turn_right → final_center → verified

Direction detection uses MediaPipe Face Mesh (468 landmarks) to compute the
horizontal yaw angle from nose tip + ear landmark positions. This is far more
reliable than Haar cascades on front-facing phone cameras, which rarely
produce a clean enough profile to trigger haarcascade_profileface.xml.

Yaw thresholds (tunable at the top of this file):
    CENTER_MAX_YAW  — degrees either side of 0 that count as "center"
    TURN_MIN_YAW    — degrees a face must rotate to count as "left" or "right"

Model path: backend/models/head_movement.pth  (relative to uvicorn launch dir)
"""

import threading
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
from torchvision import models, transforms


# ─── Tunable thresholds ───────────────────────────────────────────────────────

# Face must be within ±CENTER_MAX_YAW degrees to be "center"
CENTER_MAX_YAW = 12.0

# Face must be rotated at least TURN_MIN_YAW degrees to be "left" or "right"
TURN_MIN_YAW = 20.0


# ─── Device ───────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Liveness Model ───────────────────────────────────────────────────────────

liveness_model = models.mobilenet_v3_small(weights=None)
num_features = liveness_model.classifier[3].in_features
liveness_model.classifier[3] = nn.Linear(num_features, 2)

MODEL_PATH = "models/head_movement.pth"

try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    liveness_model.load_state_dict(checkpoint)
    liveness_model = liveness_model.to(device).eval()
    print("✅ Head-movement / liveness model loaded")
except Exception as e:
    print("❌ Model load failed:", e)
    liveness_model = None


# ─── Preprocessing ────────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─── MediaPipe Face Mesh ──────────────────────────────────────────────────────

_mp_face_mesh = mp.solutions.face_mesh
_face_mesh = _mp_face_mesh.FaceMesh(
    static_image_mode=True,   # each frame is independent — no tracking state
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
)

# Landmark indices used for yaw estimation
# Nose tip: 1   Left ear tragion: 234   Right ear tragion: 454
NOSE_TIP      = 1
LEFT_EAR      = 234
RIGHT_EAR     = 454


# ─── Per-session State Store ──────────────────────────────────────────────────

_state_lock     = threading.Lock()
_session_states: dict[str, dict] = {}

STAGE_ORDER = [
    "look_straight",
    "turn_left",
    "center",
    "turn_right",
    "final_center",
    "verified",
]


def _get_state(session_id: str) -> dict:
    with _state_lock:
        if session_id not in _session_states:
            _session_states[session_id] = {"stage": "look_straight", "completed": False}
        return _session_states[session_id]


def _set_stage(session_id: str, stage: str) -> None:
    with _state_lock:
        if session_id in _session_states:
            _session_states[session_id]["stage"] = stage


def reset_session(session_id: str) -> None:
    with _state_lock:
        _session_states[session_id] = {"stage": "look_straight", "completed": False}


# ─── Direction Detection (MediaPipe yaw) ──────────────────────────────────────

def _compute_yaw_deg(landmarks, w: int, h: int) -> float | None:
    """
    Estimate horizontal head yaw from face mesh landmarks.

    Returns signed yaw in degrees:
        negative → face turned LEFT  (nose moves left on screen)
        positive → face turned RIGHT (nose moves right on screen)
        None     → no face detected

    Method: compare the horizontal distance from nose to each ear.
    When the face is centred both distances are equal.
    When the face turns left the left-ear distance shrinks and the
    right-ear distance grows, and vice versa.

    ratio = (right_dist - left_dist) / (right_dist + left_dist)
    scaled to degrees by a calibration factor (~45 maps ratio 1.0 → 45 °).
    This is a stable heuristic that works well on front-facing cameras.
    """
    nose  = landmarks[NOSE_TIP]
    l_ear = landmarks[LEFT_EAR]
    r_ear = landmarks[RIGHT_EAR]

    nx, ny = nose.x * w,  nose.y * h
    lx, ly = l_ear.x * w, l_ear.y * h
    rx, ry = r_ear.x * w, r_ear.y * h

    left_dist  = np.sqrt((nx - lx) ** 2 + (ny - ly) ** 2)
    right_dist = np.sqrt((nx - rx) ** 2 + (ny - ry) ** 2)
    total      = left_dist + right_dist

    if total < 1e-6:
        return None

    ratio = (right_dist - left_dist) / total   # –1 … +1
    yaw   = ratio * 45.0                        # scale to degrees
    return yaw


def detect_direction(frame_bgr) -> str:
    """
    Returns "center" | "left" | "right" | "unknown".

    Uses MediaPipe Face Mesh yaw estimation — reliable on selfie cameras.
    """
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = _face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return "unknown"

    landmarks = results.multi_face_landmarks[0].landmark
    yaw = _compute_yaw_deg(landmarks, w, h)

    if yaw is None:
        return "unknown"

    print(f"  yaw={yaw:+.1f}°  (center<±{CENTER_MAX_YAW}, turn>±{TURN_MIN_YAW})")

    if abs(yaw) <= CENTER_MAX_YAW:
        return "center"
    if yaw < -TURN_MIN_YAW:
        return "left"
    if yaw > TURN_MIN_YAW:
        return "right"

    # In the dead-zone between thresholds — not turned far enough yet
    return "unknown"


# ─── Liveness Check ───────────────────────────────────────────────────────────

def run_liveness(frame) -> tuple[bool, float]:
    """
    Runs the MobileNetV3-Small liveness classifier on a BGR frame.
    Returns (is_live: bool, confidence: float 0–100).
    class 1 = LIVE, class 0 = SPOOF
    """
    rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs       = liveness_model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        _, prediction = torch.max(outputs, 1)

    predicted_class = prediction.item()
    confidence      = probabilities[predicted_class].item() * 100

    print(f"  Liveness — class: {predicted_class}, confidence: {confidence:.1f}%")
    return predicted_class == 1, confidence


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def detect_head_movement(frame, session_id: str) -> dict:
    """
    Advance the per-session head-movement state machine by one step.

    Returns HeadMovementResponse-compatible dict:
        { success, stage, message, confidence? }

    success=True only when stage reaches "verified" and liveness passes.
    All intermediate transitions return success=False with the next stage.
    """
    if liveness_model is None:
        return {"success": False, "stage": "look_straight", "message": "Model not loaded"}

    try:
        direction = detect_direction(frame)
        state     = _get_state(session_id)
        stage     = state["stage"]

        print(f"[{session_id[:8]}] stage={stage!r:14} direction={direction!r}")

        # ── look_straight ──────────────────────────────────────────────────
        if stage == "look_straight":
            if direction == "center":
                _set_stage(session_id, "turn_left")
                return {"success": False, "stage": "turn_left", "message": "Turn your head LEFT"}

        # ── turn_left ──────────────────────────────────────────────────────
        elif stage == "turn_left":
            if direction == "left":
                _set_stage(session_id, "center")
                return {"success": False, "stage": "center", "message": "Return to center"}

        # ── center ─────────────────────────────────────────────────────────
        elif stage == "center":
            if direction == "center":
                _set_stage(session_id, "turn_right")
                return {"success": False, "stage": "turn_right", "message": "Turn your head RIGHT"}

        # ── turn_right ─────────────────────────────────────────────────────
        elif stage == "turn_right":
            if direction == "right":
                _set_stage(session_id, "final_center")
                return {"success": False, "stage": "final_center", "message": "Look straight again"}

        # ── final_center — run liveness ────────────────────────────────────
        elif stage == "final_center":
            if direction == "center":
                live, confidence = run_liveness(frame)
                if live:
                    with _state_lock:
                        _session_states[session_id]["stage"]     = "verified"
                        _session_states[session_id]["completed"] = True
                    return {"success": True, "stage": "verified", "message": "Liveness verified", "confidence": confidence}
                else:
                    reset_session(session_id)
                    return {"success": False, "stage": "look_straight", "message": "Spoof detected — please try again"}

        # Fallback: direction not yet matched for current stage
        return {"success": False, "stage": stage, "message": "Waiting for correct movement"}

    except Exception as e:
        print("Inference error:", e)
        return {"success": False, "stage": "look_straight", "message": str(e)}