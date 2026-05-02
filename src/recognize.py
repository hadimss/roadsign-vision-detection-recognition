"""
Traffic sign recognition module.

This module loads the fine-tuned ResNet-18 classifier and predicts
the exact traffic sign class from a cropped traffic sign image.
"""

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


CLASS_NAMES = [
    "Speed limit 20 km/h",
    "Speed limit 30 km/h",
    "Speed limit 50 km/h",
    "Speed limit 60 km/h",
    "Speed limit 70 km/h",
    "Speed limit 80 km/h",
    "End of speed limit 80 km/h",
    "Speed limit 100 km/h",
    "Speed limit 120 km/h",
    "No passing",
    "No passing for vehicles over 3.5 tons",
    "Right-of-way at next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve left",
    "Dangerous curve right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing for vehicles over 3.5 tons",
]


def get_device() -> torch.device:
    """
    Use CPU for stable inference across systems.
    """
    return torch.device("cpu")


def build_classifier(num_classes: int = 43) -> nn.Module:
    """
    Build ResNet-18 classifier with 43 output classes.
    """
    model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_recognition_transform(image_size: int = 224):
    """
    Preprocessing used before classification.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_classifier(
    model_path: str = "models/best_resnet18_gtsrb.pth",
    device: torch.device | None = None,
) -> nn.Module:
    """
    Load the trained ResNet-18 traffic sign classifier.
    """

    if device is None:
        device = get_device()

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {model_path}. "
            "Copy the trained classifier from the classification project first."
        )

    checkpoint = torch.load(model_path, map_location=device)

    model = build_classifier(num_classes=43)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def recognize_crop(
    image: Image.Image,
    classifier: nn.Module,
    device: torch.device | None = None,
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """
    Predict the traffic sign class from a cropped sign image.

    Returns:
        List of top-k predictions as (class_name, confidence).
    """

    if device is None:
        device = get_device()

    image = image.convert("RGB")
    transform = get_recognition_transform()

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = classifier(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k)

    predictions = []

    for probability, class_index in zip(top_probs[0], top_indices[0]):
        class_name = CLASS_NAMES[class_index.item()]
        confidence = probability.item()
        predictions.append((class_name, confidence))

    return predictions


if __name__ == "__main__":
    device = get_device()
    classifier = load_classifier(device=device)

    print("Classifier loaded successfully.")
    print(f"Device: {device}")
    print(f"Number of classes: {len(CLASS_NAMES)}")