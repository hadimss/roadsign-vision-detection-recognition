"""
Train a YOLO object detection model on the prepared GTSDB dataset.
"""

from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)


def get_device2():
    if torch.cuda.is_available():
        return 0

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"

def get_device():
    """
    Use CPU for YOLO training on macOS because torchvision NMS is not fully
    supported on MPS during validation.
    """
    return "cpu"


def main():
    config = load_config()

    detection_config = config["detection"]

    processed_data_dir = ROOT_DIR / config["paths"]["processed_data_dir"]
    model_dir = ROOT_DIR / config["paths"]["model_dir"]

    data_yaml_path = processed_data_dir / "data.yaml"

    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"YOLO data file not found: {data_yaml_path}. "
            "Run src/prepare_dataset.py first."
        )

    model_dir.mkdir(parents=True, exist_ok=True)

    model_name = detection_config["model_name"]
    image_size = detection_config["image_size"]
    epochs = detection_config["epochs"]
    batch_size = detection_config["batch_size"]
    device = get_device()

    print(f"Using device: {device}")
    print(f"Loading YOLO model: {model_name}")

    model = YOLO(model_name)

    print("Starting YOLO training...")

    model.train(
        data=str(data_yaml_path.resolve()),
        imgsz=image_size,
        epochs=epochs,
        batch=batch_size,
        project=str(model_dir.resolve()),
        name="yolo_gtsdb",
        exist_ok=True,
        device=device,
    )

    print("Training complete.")
    print("Best model saved under:")
    print(model_dir / "yolo_gtsdb" / "weights" / "best.pt")


if __name__ == "__main__":
    main()