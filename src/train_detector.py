"""
Train a YOLO object detection model on the prepared GTSDB dataset.
"""

from pathlib import Path
import yaml
from ultralytics import YOLO


CONFIG_PATH = Path("configs/config.yaml")


def load_config():
    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()

    detection_config = config["detection"]
    processed_data_dir = Path(config["paths"]["processed_data_dir"])

    data_yaml_path = processed_data_dir / "data.yaml"

    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"YOLO data file not found: {data_yaml_path}. "
            "Run src/prepare_dataset.py first."
        )

    model_name = detection_config["model_name"]
    image_size = detection_config["image_size"]
    epochs = detection_config["epochs"]
    batch_size = detection_config["batch_size"]

    print(f"Loading YOLO model: {model_name}")
    model = YOLO(model_name)

    print("Starting YOLO training...")
    model.train(
        data=str(data_yaml_path),
        imgsz=image_size,
        epochs=epochs,
        batch=batch_size,
        project="models",
        name="yolo_gtsdb",
        exist_ok=True,
    )

    print("Training complete.")
    print("Best model should be saved under:")
    print("models/yolo_gtsdb/weights/best.pt")


if __name__ == "__main__":
    main()