"""
Prepare the GTSDB dataset for YOLO training.

This script converts the original GTSDB annotation format:

filename;left;top;right;bottom;class_id

into YOLO format:

class_id x_center y_center width height

All bounding box coordinates are normalized between 0 and 1.
"""

from pathlib import Path
from collections import defaultdict
from PIL import Image
import shutil
import yaml


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


RAW_DATA_DIR = Path("data/raw/FullIJCNN2013")
PROCESSED_DATA_DIR = Path("data/processed")


def read_annotations(annotation_file: Path):
    """
    Read GTSDB annotation file and group boxes by image filename.
    """

    annotations = defaultdict(list)

    with open(annotation_file, "r") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            parts = line.split(";")

            if len(parts) != 6:
                continue

            filename = parts[0]
            left = int(parts[1])
            top = int(parts[2])
            right = int(parts[3])
            bottom = int(parts[4])
            class_id = int(parts[5])

            annotations[filename].append(
                {
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "class_id": class_id,
                }
            )

    return annotations


def convert_bbox_to_yolo(box, image_width, image_height):
    """
    Convert one bounding box from pixel coordinates to YOLO format.
    """

    left = box["left"]
    top = box["top"]
    right = box["right"]
    bottom = box["bottom"]

    x_center = ((left + right) / 2) / image_width
    y_center = ((top + bottom) / 2) / image_height
    width = (right - left) / image_width
    height = (bottom - top) / image_height

    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    width = min(max(width, 0.0), 1.0)
    height = min(max(height, 0.0), 1.0)

    return x_center, y_center, width, height


def get_split(filename: str):
    """
    Use the official-style split:
    images 00000-00599 for training
    images 00600-00899 for validation
    """

    image_id = int(Path(filename).stem)

    if image_id < 600:
        return "train"

    return "val"


def prepare_directories():
    """
    Create YOLO dataset directories.
    """

    for split in ["train", "val"]:
        (PROCESSED_DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (PROCESSED_DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def convert_dataset():
    """
    Convert GTSDB dataset into YOLO format.
    """

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Raw dataset folder not found: {RAW_DATA_DIR}. "
            "Please download and unzip FullIJCNN2013 first."
        )

    annotation_file = RAW_DATA_DIR / "gt.txt"

    if not annotation_file.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_file}"
        )

    prepare_directories()

    annotations = read_annotations(annotation_file)
    image_paths = sorted(RAW_DATA_DIR.glob("*.ppm"))

    if not image_paths:
        raise FileNotFoundError(
            f"No .ppm images found in {RAW_DATA_DIR}"
        )

    total_images = 0
    total_boxes = 0

    for image_path in image_paths:
        filename = image_path.name
        split = get_split(filename)

        output_image_path = (
            PROCESSED_DATA_DIR / "images" / split / f"{image_path.stem}.jpg"
        )

        output_label_path = (
            PROCESSED_DATA_DIR / "labels" / split / f"{image_path.stem}.txt"
        )

        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        image.save(output_image_path, quality=95)

        boxes = annotations.get(filename, [])

        with open(output_label_path, "w") as label_file:
            for box in boxes:
                class_id = 0
                x_center, y_center, width, height = convert_bbox_to_yolo(
                    box,
                    image_width,
                    image_height,
                )

                label_file.write(
                    f"{class_id} {x_center:.6f} {y_center:.6f} "
                    f"{width:.6f} {height:.6f}\n"
                )

        total_images += 1
        total_boxes += len(boxes)

    data_yaml = {
    "path": str(PROCESSED_DATA_DIR.resolve()),
    "train": "images/train",
    "val": "images/val",
    "nc": 1,
    "names": ["traffic sign"],
    }

    data_yaml_path = PROCESSED_DATA_DIR / "data.yaml"

    with open(data_yaml_path, "w") as file:
        yaml.safe_dump(data_yaml, file, sort_keys=False)

    print("Dataset conversion complete.")
    print(f"Processed images: {total_images}")
    print(f"Processed bounding boxes: {total_boxes}")
    print(f"YOLO config saved to: {data_yaml_path}")


if __name__ == "__main__":
    convert_dataset()