"""
End-to-end detection and recognition pipeline.

This module will combine:
1. YOLO-based traffic sign detection
2. Cropping detected signs
3. Recognition/classification of each detected sign
4. Annotated output visualization
"""

from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


def load_detector(model_path: str):
    """
    Load YOLO detector.
    """
    return YOLO(model_path)


def crop_detection(image: np.ndarray, box_xyxy) -> Image.Image:
    """
    Crop one detected bounding box from an image.

    Args:
        image: Original image as a NumPy array.
        box_xyxy: Bounding box in [x1, y1, x2, y2] format.

    Returns:
        PIL image crop.
    """

    x1, y1, x2, y2 = map(int, box_xyxy)

    height, width = image.shape[:2]

    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    crop = image[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    return Image.fromarray(crop_rgb)


def run_detection(
    image_path: str,
    detector,
    confidence_threshold: float = 0.25,
) -> List[Dict]:
    """
    Run detection and return bounding boxes and cropped signs.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    results = detector.predict(
        source=str(image_path),
        conf=confidence_threshold,
        save=False,
        verbose=False,
    )

    result = results[0]
    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf[0].cpu().numpy())
        class_id = int(box.cls[0].cpu().numpy())

        crop = crop_detection(image, xyxy)

        detections.append(
            {
                "bbox": xyxy.tolist(),
                "confidence": confidence,
                "detector_class_id": class_id,
                "crop": crop,
            }
        )

    return detections


def annotate_image(
    image_path: str,
    detections: List[Dict],
    output_path: str,
):
    """
    Draw detection boxes on the original image.
    """

    image = cv2.imread(str(image_path))

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["bbox"])
        confidence = detection["confidence"]

        label = f"traffic sign {confidence:.2f}"

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), image)

    return output_path