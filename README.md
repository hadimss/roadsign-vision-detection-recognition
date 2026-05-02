# RoadSign Vision: Traffic Sign Detection and Recognition

An advanced computer vision project for detecting and recognizing traffic signs in full road images.

This project extends traffic sign classification into a more realistic driver-assistance scenario. Instead of assuming that the traffic sign is already cropped, the system detects traffic signs in full road images, draws bounding boxes around them, and recognizes the detected sign class.

---

## Project Overview

Traffic sign recognition is an important component of driver-assistance and autonomous driving systems. A real-world system should be able to locate traffic signs in full road scenes and identify what each sign means.

This project focuses on building an end-to-end pipeline for:

```text
Full road image
→ Traffic sign detection
→ Bounding box prediction
→ Traffic sign recognition
→ Visual output with labels and confidence scores
```

---

## Problem Statement

A basic traffic sign classifier can recognize a sign only when the image is already cropped around the sign.

However, in real-world driving scenes, traffic signs appear:

- at different distances
- under different lighting conditions
- at different scales
- with different backgrounds
- sometimes partially occluded

This project addresses the more realistic task of detecting and recognizing traffic signs directly from full road images.

---

## Project Objectives

- Prepare a traffic sign detection dataset with bounding box annotations.
- Convert annotations into YOLO format.
- Fine-tune a YOLO object detection model.
- Evaluate detection performance using object detection metrics.
- Connect detection with recognition/classification.
- Build a Streamlit application for image upload and visual prediction.
- Display detected traffic signs with bounding boxes, labels, and confidence scores.

---

## Dataset

This project uses the **German Traffic Sign Detection Benchmark**, also known as **GTSDB**.

The dataset contains full road images with traffic sign bounding box annotations.

The raw dataset is downloaded locally into:

```text
data/raw/
```

The processed YOLO-format dataset is saved into:

```text
data/processed/
```

Dataset files are not uploaded to GitHub because they are large.

---

## Planned Dataset Format

The processed dataset will follow the YOLO format:

```text
data/processed/
│
├── images/
│   ├── train/
│   └── val/
│
├── labels/
│   ├── train/
│   └── val/
│
└── data.yaml
```

Each label file will contain bounding boxes in this format:

```text
class_id x_center y_center width height
```

All bounding box values are normalized between `0` and `1`.

---

## Model Architecture

The project uses a two-stage design:

```text
Stage 1: YOLO detector
Full road image → traffic sign bounding boxes

Stage 2: Recognition model
Detected traffic sign crop → traffic sign class
```

### Detection Model

The detection model is based on YOLO.

Initial model:

```text
YOLOv8 Nano
```

This model is lightweight and suitable for experimentation on a laptop.

### Recognition Model

The recognition stage can use a fine-tuned ResNet-18 classifier trained on cropped traffic sign images.

This creates a complete detection + recognition pipeline:

```text
Road image
→ YOLO detects traffic sign
→ Crop detected sign
→ ResNet-18 recognizes sign class
→ App displays result
```

---

## Project Pipeline

```text
1. Download GTSDB dataset
2. Parse original bounding box annotations
3. Convert annotations to YOLO format
4. Train YOLO detector
5. Evaluate detector performance
6. Run inference on road images
7. Crop detected traffic signs
8. Recognize detected signs
9. Display results in Streamlit
```

---

## Project Structure

```text
roadsign-vision-detection-recognition/
│
├── app/
│   └── streamlit_app.py
│
├── configs/
│   └── config.yaml
│
├── data/
│   └── README.md
│
├── docs/
│   ├── DATASET.md
│   ├── MODEL_CARD.md
│   └── PIPELINE.md
│
├── models/
│   └── README.md
│
├── notebooks/
│   └── .gitkeep
│
├── reports/
│   └── figures/
│
├── src/
│   ├── prepare_dataset.py
│   ├── train_detector.py
│   ├── detect.py
│   ├── recognize.py
│   ├── pipeline.py
│   └── utils.py
│
├── tests/
│   └── test_pipeline.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/hadimss/roadsign-vision-detection-recognition.git
cd roadsign-vision-detection-recognition
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

After downloading and extracting the dataset, prepare it for YOLO training:

```bash
python3 src/prepare_dataset.py
```

This creates:

```text
data/processed/images/train/
data/processed/images/val/
data/processed/labels/train/
data/processed/labels/val/
data/processed/data.yaml
```

---

## Training

Train the YOLO detector:

```bash
python3 src/train_detector.py
```

The best detector checkpoint will be saved locally under:

```text
models/yolo_gtsdb/weights/best.pt
```

Model checkpoints are ignored by Git because they are large.

---

## Application

The final application will allow a user to upload a road image and receive:

- detected traffic sign locations
- bounding boxes
- predicted class labels
- confidence scores

The app will be run using:

```bash
streamlit run app/streamlit_app.py
```

---

## Technologies Used

- Python
- PyTorch
- TorchVision
- Ultralytics YOLO
- OpenCV
- Pillow
- NumPy
- Pandas
- Matplotlib
- Streamlit

---

## Current Status

Project setup and documentation are in progress.

Completed:

- Repository structure
- Configuration file
- Project documentation
- Dataset download in progress
- YOLO dataset preparation script
- YOLO training script

Next steps:

- Finish dataset download
- Convert GTSDB annotations to YOLO format
- Train the detection model
- Evaluate detection performance
- Build the detection + recognition demo app

---

## Future Improvements

- Add real-time webcam or video inference.
- Compare YOLOv8n with larger YOLO models.
- Add two-stage recognition using a fine-tuned ResNet classifier.
- Deploy the Streamlit app online.
- Export the detector to ONNX for lightweight deployment.
- Add support for multiple traffic signs in one image.

---

## Author

**Hadi Al Masri**