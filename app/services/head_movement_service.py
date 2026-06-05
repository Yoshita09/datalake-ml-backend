import cv2
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms


# Device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Build Architecture

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


# Preprocessing

transform = transforms.Compose([
    transforms.ToPILImage(),

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


# Inference

def detect_head_movement(frame):

    if liveness_model is None:

        return {
            "success": False,
            "message": "Model not loaded",
        }

    try:

        # BGR → RGB
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        # preprocess
        input_tensor = transform(rgb)

        # batch dimension
        input_tensor = (
            input_tensor
            .unsqueeze(0)
            .to(device)
        )

        # inference
        with torch.no_grad():

            outputs = liveness_model(
                input_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )[0]

            _, prediction = torch.max(
                outputs,
                1,
            )

        confidence = (
            probabilities[
                prediction.item()
            ].item()
            * 100
        )

        predicted_class = prediction.item()

        print(
            "Prediction:",
            predicted_class,
            "Confidence:",
            confidence,
        )

        # notebook logic:
        # 0 = live
        # 1 = spoof

        success = predicted_class == 0

        return {
            "success": success,
            "message": (
                "Live user verified"
                if success
                else "Spoof detected"
            ),
            "confidence": confidence,
        }

    except Exception as e:

        print("Inference error:", e)

        return {
            "success": False,
            "message": str(e),
        }