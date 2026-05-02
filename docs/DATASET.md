# Dataset Documentation

## Dataset Name

German Traffic Sign Detection Benchmark, also known as GTSDB.

## Dataset Purpose

The dataset is used for traffic sign detection in full road images. Unlike cropped classification datasets, this dataset contains complete road scenes with bounding box annotations around traffic signs.

## Raw Data

Raw data is stored locally in:

```text
data/raw/
```

This folder is ignored by Git.

Expected raw dataset structure after extraction:

```text
data/raw/FullIJCNN2013/
│
├── 00000.ppm
├── 00001.ppm
├── ...
└── gt.txt
```

## Annotation Format

The original annotation file uses this format:

```text
filename;left;top;right;bottom;class_id
```

Example:

```text
00000.ppm;774;411;815;446;11
```

Where:

| Field | Meaning |
|---|---|
| filename | Image file name |
| left | Left x-coordinate of bounding box |
| top | Top y-coordinate of bounding box |
| right | Right x-coordinate of bounding box |
| bottom | Bottom y-coordinate of bounding box |
| class_id | Traffic sign class ID |

## YOLO Format

YOLO requires one label file per image.

Each label file contains:

```text
class_id x_center y_center width height
```

All bounding box values are normalized between `0` and `1`.

## Processed Data

Processed data is saved locally in:

```text
data/processed/
```

Expected structure:

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

## Notes

Dataset files are not committed to GitHub because they are large. Users should download and prepare the dataset locally.