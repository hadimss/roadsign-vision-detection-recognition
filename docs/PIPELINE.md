# Project Pipeline

This document describes the full detection and recognition workflow.

---

## 1. Input

The system takes a full road image as input.

Example:

```text
road_scene.jpg
```

---

## 2. Detection

A YOLO object detection model processes the image and predicts traffic sign locations.

Output:

```text
bounding box coordinates
confidence score
detected object class
```

---

## 3. Cropping

Each detected traffic sign region is cropped from the original image.

This crop can be passed to a recognition model.

---

## 4. Recognition

The recognition model classifies the cropped sign image into one of the traffic sign classes.

Example output:

```text
Speed limit 50 km/h
Stop
Yield
No entry
```

---

## 5. Visualization

The final system draws bounding boxes on the original road image.

Each box displays:

```text
class label
confidence score
```

---

## 6. Streamlit App

The final application will allow users to upload an image and view the detection results interactively.

Expected user flow:

```text
Upload image
→ Run detection
→ Show image with bounding boxes
→ Show detected signs and confidence scores
```

---

## 7. End-to-End Workflow

```text
Road image
→ YOLO detector
→ Bounding boxes
→ Crop detected signs
→ Recognition model
→ Final annotated image
```

---

## Current Implementation Status

Completed:

- Project structure
- Dataset preparation script
- YOLO training script

Pending:

- Dataset conversion
- YOLO training
- Detection inference script
- Recognition integration
- Streamlit app