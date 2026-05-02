"""
Run traffic sign detection on an image using a trained YOLO model.

This script loads a YOLO detector, runs inference on an input image,
and saves the annotated image with bounding boxes.
"""

from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO


CONFIG_PATH = Path("configs/config.yaml")


def load_config():
    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)


def get_detector_model_path(config: dict) -> Path:
    """
    Return the trained YOLO model path if available.
    Otherwise, return the base YOLO model from the config.
    """

    trained_model_path = Path("models/yolo_gtsdb/weights/best.pt")

    if trained_model_path.exists():
        return trained_model_path

    return Path(config["detection"]["model_name"])


def detect_image(
    image_path: str,
    output_path: str = "reports/figures/detection_result.jpg",
    confidence_threshold: float = 0.25,
):
    """
    Run YOLO detection on a single image.

    Args:
        image_path: Path to input image.
        output_path: Path where annotated image will be saved.
        confidence_threshold: Minimum confidence for detections.
    """

    config = load_config()
    model_path = get_detector_model_path(config)

    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    print(f"Loading YOLO model: {model_path}")
    model = YOLO(str(model_path))

    print(f"Running detection on: {image_path}")
    results = model.predict(
        source=str(image_path),
        conf=confidence_threshold,
        save=False,
        verbose=False,
    )

    result = results[0]
    annotated_image = result.plot()

    # result.plot() returns RGB image, OpenCV saves BGR.
    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), annotated_image_bgr)

    print(f"Detection result saved to: {output_path}")

    if result.boxes is not None:
        print(f"Number of detections: {len(result.boxes)}")
    else:
        print("Number of detections: 0")


if __name__ == "__main__":
    # Example usage after dataset is extracted:
    # python3 src/detect.py
    example_image = "data/processed/images/train/00000.jpg"

    detect_image(
        image_path=example_image,
        output_path="reports/figures/detection_result.jpg",
    )