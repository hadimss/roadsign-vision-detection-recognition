"""
End-to-end traffic sign detection and recognition pipeline.

Pipeline:
1. Load full road image
2. Detect traffic signs using YOLO
3. Crop each detected sign
4. Recognize each crop using the ResNet-18 classifier
5. Save annotated image with exact traffic sign labels
"""

import argparse
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from recognize import get_device, load_classifier, recognize_crop


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_DETECTOR_PATH = ROOT_DIR / "models" / "yolo_gtsdb" / "weights" / "best.pt"
DEFAULT_CLASSIFIER_PATH = ROOT_DIR / "models" / "best_resnet18_gtsrb.pth"


def load_detector(model_path: str | Path = DEFAULT_DETECTOR_PATH):
    """
    Load YOLO detector.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Detector model not found: {model_path}. "
            "Train the YOLO detector first."
        )

    return YOLO(str(model_path))


def crop_detection(image_bgr: np.ndarray, box_xyxy) -> Image.Image:
    """
    Crop one detected bounding box from the original image.
    """

    x1, y1, x2, y2 = map(int, box_xyxy)

    height, width = image_bgr.shape[:2]

    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    crop_bgr = image_bgr[y1:y2, x1:x2]

    if crop_bgr.size == 0:
        raise ValueError("Empty crop generated from detection box.")

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(crop_rgb)


def run_detection_and_recognition(
    image_path: str | Path,
    detector,
    classifier,
    confidence_threshold: float = 0.25,
) -> List[Dict]:
    """
    Run YOLO detection and classify each detected crop.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    results = detector.predict(
        source=str(image_path),
        conf=confidence_threshold,
        save=False,
        verbose=False,
        device="cpu",
    )

    result = results[0]
    detections = []

    if result.boxes is None:
        return detections

    device = get_device()

    for box in result.boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        detection_confidence = float(box.conf[0].cpu().numpy())

        crop = crop_detection(image_bgr, xyxy)

        recognition_predictions = recognize_crop(
            image=crop,
            classifier=classifier,
            device=device,
            top_k=3,
        )

        top_class_name, top_class_confidence = recognition_predictions[0]

        detections.append(
            {
                "bbox": xyxy.tolist(),
                "detection_confidence": detection_confidence,
                "recognized_class": top_class_name,
                "recognition_confidence": top_class_confidence,
                "top_3_predictions": recognition_predictions,
                "crop": crop,
            }
        )

    return detections


def annotate_image(
    image_path: str | Path,
    detections: List[Dict],
    output_path: str | Path,
):
    """
    Draw bounding boxes and recognized class names on the image.
    """

    image_path = Path(image_path)
    output_path = Path(output_path)

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["bbox"])

        class_name = detection["recognized_class"]
        det_conf = detection["detection_confidence"]
        cls_conf = detection["recognition_confidence"]

        label = f"{class_name} | det {det_conf:.2f} | cls {cls_conf:.2f}"

        cv2.rectangle(
            image_bgr,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2,
        )

        cv2.putText(
            image_bgr,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image_bgr)

    return output_path


def run_pipeline(
    image_path: str | Path,
    output_path: str | Path = "reports/figures/detection_recognition_result.jpg",
    confidence_threshold: float = 0.25,
):
    """
    Run the full detection + recognition pipeline.
    """

    detector = load_detector(DEFAULT_DETECTOR_PATH)
    classifier = load_classifier(DEFAULT_CLASSIFIER_PATH)

    detections = run_detection_and_recognition(
        image_path=image_path,
        detector=detector,
        classifier=classifier,
        confidence_threshold=confidence_threshold,
    )

    output_path = annotate_image(
        image_path=image_path,
        detections=detections,
        output_path=output_path,
    )

    print("Pipeline complete.")
    print(f"Input image: {image_path}")
    print(f"Output image: {output_path}")
    print(f"Number of detections: {len(detections)}")

    for i, detection in enumerate(detections, start=1):
        print(f"\nDetection {i}")
        print(f"Detected class: {detection['recognized_class']}")
        print(f"Detection confidence: {detection['detection_confidence']:.4f}")
        print(f"Recognition confidence: {detection['recognition_confidence']:.4f}")

        print("Top 3 recognition predictions:")
        for class_name, confidence in detection["top_3_predictions"]:
            print(f"  - {class_name}: {confidence:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run traffic sign detection and recognition."
    )

    parser.add_argument(
        "--image",
        type=str,
        default="data/processed/images/train/00000.jpg",
        help="Path to input road image.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="reports/figures/detection_recognition_result.jpg",
        help="Path to save annotated output image.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO detection confidence threshold.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_pipeline(
        image_path=args.image,
        output_path=args.output,
        confidence_threshold=args.conf,
    )