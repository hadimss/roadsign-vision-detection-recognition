# Model Card

## Project Name

RoadSign Vision: Traffic Sign Detection and Recognition

## Model Type

Object detection and image classification pipeline.

## Detection Model

Initial detection model:

```text
YOLOv8 Nano
```

The detector is used to locate traffic signs in full road images.

## Recognition Model

The recognition stage can use a fine-tuned ResNet-18 classifier for cropped traffic sign classification.

## Intended Use

This project is intended for educational and research purposes.

It demonstrates how traffic sign detection and recognition can be combined in a driver-assistance style computer vision pipeline.

## Not Intended For

This project is not intended for real-world driving decisions or safety-critical autonomous vehicle deployment.

## Input

Full road image.

Supported image formats may include:

```text
.jpg
.jpeg
.png
.ppm
```

## Output

The model pipeline produces:

```text
bounding boxes
traffic sign labels
confidence scores
annotated output image
```

## Limitations

Possible limitations include:

- small dataset size
- limited traffic sign types
- signs from a specific geographic region
- performance may decrease under poor lighting
- performance may decrease with motion blur
- model may not generalize to all countries or road environments

## Future Improvements

- Train with larger and more diverse datasets.
- Add video and webcam support.
- Compare different YOLO model sizes.
- Add deployment support using ONNX.
- Improve recognition with a stronger classifier or transformer model.