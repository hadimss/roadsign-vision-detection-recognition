"""
Streamlit app for traffic sign detection and recognition.

The app:
1. Uploads a road image
2. Detects traffic signs using YOLO
3. Crops detected signs
4. Recognizes exact traffic sign classes using ResNet-18
5. Displays annotated output and top predictions
"""

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from pipeline import (
    DEFAULT_CLASSIFIER_PATH,
    DEFAULT_DETECTOR_PATH,
    annotate_image,
    load_detector,
    run_detection_and_recognition,
)
from recognize import load_classifier


UPLOAD_DIR = ROOT_DIR / "reports" / "figures" / "app_uploads"
OUTPUT_DIR = ROOT_DIR / "reports" / "figures"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="RoadSign Vision",
    page_icon="🚦",
    layout="wide",
)


@st.cache_resource
def load_models():
    detector = load_detector(DEFAULT_DETECTOR_PATH)
    classifier = load_classifier(DEFAULT_CLASSIFIER_PATH)
    return detector, classifier


st.title("🚦 RoadSign Vision")
st.subheader("Traffic Sign Detection and Recognition")

st.write(
    "Upload a road image. The system will detect traffic signs, crop them, "
    "recognize the exact sign class, and display the results."
)

st.markdown("---")

with st.sidebar:
    st.header("Settings")

    confidence_threshold = st.slider(
        "Detection confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
    )

    st.markdown("### Required local models")

    detector_exists = DEFAULT_DETECTOR_PATH.exists()
    classifier_exists = DEFAULT_CLASSIFIER_PATH.exists()

    st.write(f"YOLO detector: {'✅ Found' if detector_exists else '❌ Missing'}")
    st.write(f"ResNet classifier: {'✅ Found' if classifier_exists else '❌ Missing'}")

    st.caption("Model files are stored locally and are ignored by Git.")


uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png", "bmp"],
)

if uploaded_file is None:
    st.info("Upload an image to start detection and recognition.")

else:
    input_image = Image.open(uploaded_file).convert("RGB")

    input_path = UPLOAD_DIR / "uploaded_road_image.jpg"
    output_path = OUTPUT_DIR / "app_detection_recognition_result.jpg"

    input_image.save(input_path)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(input_image, use_container_width=True)

    try:
        with st.spinner("Loading models and running detection + recognition..."):
            detector, classifier = load_models()

            detections = run_detection_and_recognition(
                image_path=input_path,
                detector=detector,
                classifier=classifier,
                confidence_threshold=confidence_threshold,
            )

            annotate_image(
                image_path=input_path,
                detections=detections,
                output_path=output_path,
            )

        with col2:
            st.subheader("Annotated Result")
            st.image(str(output_path), use_container_width=True)

        st.markdown("---")

        st.subheader("Detection Results")

        if len(detections) == 0:
            st.warning("No traffic signs were detected. Try lowering the confidence threshold.")

        else:
            st.success(f"Detected {len(detections)} traffic sign(s).")

            for index, detection in enumerate(detections, start=1):
                st.markdown(f"### Detection {index}")

                crop_col, info_col = st.columns([1, 2])

                with crop_col:
                    st.image(
                        detection["crop"],
                        caption="Detected sign crop",
                        use_container_width=True,
                    )

                with info_col:
                    st.write(f"**Recognized sign:** {detection['recognized_class']}")
                    st.write(
                        f"**Detection confidence:** "
                        f"{detection['detection_confidence'] * 100:.2f}%"
                    )
                    st.write(
                        f"**Recognition confidence:** "
                        f"{detection['recognition_confidence'] * 100:.2f}%"
                    )

                    st.write("**Top 3 recognition predictions:**")

                    for rank, (class_name, confidence) in enumerate(
                        detection["top_3_predictions"],
                        start=1,
                    ):
                        st.write(f"{rank}. {class_name} — {confidence * 100:.2f}%")
                        st.progress(float(confidence))

    except FileNotFoundError as error:
        st.error(str(error))

    except Exception as error:
        st.error("Something went wrong while running the app.")
        st.exception(error)

st.markdown("---")
st.caption(
    "Detector: YOLOv8 Nano fine-tuned on GTSDB | "
    "Recognizer: ResNet-18 fine-tuned on GTSRB"
)