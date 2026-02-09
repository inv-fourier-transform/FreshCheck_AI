import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from model_definition import build_model

# --------------------------------------------------
# Device configuration
# --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Get model path such that it works both locally and on Streamlit Cloud
# --------------------------------------------------
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / "models" / "fruits_classifier_resent50_tl.pth"

# Alternative using os.path:
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "fruits_classifier_resent50_tl.pth")

# --------------------------------------------------
# Cached objects
# --------------------------------------------------
trained_model = None
idx_to_class = None


def _load_model():
    global trained_model, idx_to_class

    if trained_model is not None and idx_to_class is not None:
        return

    trained_model = build_model(num_classes=2)
    trained_model.to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,  # Use the absolute path
        map_location=DEVICE
    )

    trained_model.load_state_dict(checkpoint["model_state_dict"])
    trained_model.eval()

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}


def predict_freshness(image_path):
    _load_model()

    # --------------------------------------------------
    # Preprocess image
    # --------------------------------------------------
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # --------------------------------------------------
    # Inference + confidence
    # --------------------------------------------------
    with torch.no_grad():
        logits = trained_model(image_tensor)

        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    predicted_label = idx_to_class[predicted_idx.item()]
    confidence_score = confidence.item()  # float between 0 and 1

    return predicted_label, confidence_score
