import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
IMG_SIZE = (224, 224)

CLASS_NAMES = ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]
CLASS_MAPPING_FILE = APP_DIR / "class_mapping.csv"

MODEL_OPTIONS = {
    "Custom CNN": {
        "file": APP_DIR / "custom_cnn_final.keras",
        "preprocessing": "cnn",
    },
    "MobileNetV2": {
        "file": APP_DIR / "mobilenetv2_final.keras",
        "preprocessing": "mobilenet",
    },
}

st.set_page_config(
    page_title="Histopathology Image Classifier",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------
# UI styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {font-size: 2.25rem; font-weight: 750; margin-bottom: 0.2rem;}
    .subtitle {color: #64748b; font-size: 1.02rem; margin-bottom: 1.2rem;}
    .result-card {
        padding: 1rem 1.2rem; border-radius: 14px;
        border: 1px solid rgba(128,128,128,.25);
        background: rgba(128,128,128,.06);
    }
    .warning {padding: .8rem 1rem; border-radius: 10px;
        background: #fff7ed; border: 1px solid #fed7aa; color: #9a3412;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🔬 Histopathology Image Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a histopathology image to view a model prediction and confidence score.</div>',
    unsafe_allow_html=True,
)

st.warning(
    "Educational/research demonstration only. This tool is not a medical device, "
    "does not provide a diagnosis, and must not be used to make medical decisions."
)

# Load class mapping if present; otherwise use the fixed order used during training.
if CLASS_MAPPING_FILE.exists():
    try:
        mapping_df = pd.read_csv(CLASS_MAPPING_FILE)
        if {"class_name", "class_index"}.issubset(mapping_df.columns):
            mapping_df = mapping_df.sort_values("class_index")
            loaded_names = mapping_df["class_name"].astype(str).tolist()
            if len(loaded_names) == 5:
                CLASS_NAMES = loaded_names
    except Exception:
        pass

@st.cache_resource
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path, compile=False)

def prepare_image(image: Image.Image) -> np.ndarray:
    """Resize and scale the image exactly as the notebook's input pipeline does."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def get_probabilities(model, image_batch: np.ndarray, preprocessing: str) -> np.ndarray:
    model_input = image_batch
    if preprocessing == "mobilenet":
        # The notebook's MobileNet pipeline maps [0,1] -> [0,255],
        # then applies MobileNetV2 preprocess_input (approximately [-1,1]).
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        model_input = preprocess_input(image_batch * 255.0)

    predictions = model.predict(model_input, verbose=0)
    predictions = np.asarray(predictions)

    if predictions.ndim == 2 and predictions.shape[0] == 1:
        predictions = predictions[0]
    else:
        predictions = predictions.reshape(-1)

    # The saved model is expected to output five softmax probabilities.
    if len(predictions) != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} output scores, but the model returned "
            f"{len(predictions)}. Check that the model and class mapping match."
        )
    return predictions

with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox("Choose model", list(MODEL_OPTIONS.keys()))
    st.caption("Select a model file that was saved by your training notebook.")
    st.divider()
    st.markdown("**Supported image formats**")
    st.write("JPG, JPEG, PNG")

uploaded_file = st.file_uploader(
    "Upload a histopathology image",
    type=["jpg", "jpeg", "png"],
    help="Choose one representative image from your dataset.",
)

if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file)
        original_image = ImageOps.exif_transpose(original_image).convert("RGB")

        left, right = st.columns([1, 1], gap="large")
        with left:
            st.subheader("Uploaded image")
            st.image(original_image, caption=uploaded_file.name, use_container_width=True)
            st.caption(f"Original image size: {original_image.width} × {original_image.height} pixels")

        with right:
            st.subheader("Prediction")
            model_path = MODEL_OPTIONS[selected_model]["file"]

            if not model_path.exists():
                st.error(f"Model file not found: `{model_path.name}`")
                st.info(
                    "Copy the saved model into the same folder as `app.py`, "
                    "or update the filename in MODEL_OPTIONS near the top of app.py."
                )
            else:
                with st.spinner(f"Loading {selected_model} and predicting..."):
                    model = load_model(str(model_path))
                    image_batch = prepare_image(original_image)
                    probabilities = get_probabilities(
                        model, image_batch, MODEL_OPTIONS[selected_model]["preprocessing"]
                    )

                predicted_index = int(np.argmax(probabilities))
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(probabilities[predicted_index]) * 100.0

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.metric("Predicted class", predicted_class)
                st.metric("Model confidence", f"{confidence:.2f}%")
                st.progress(float(np.clip(confidence / 100.0, 0.0, 1.0)))
                st.markdown("</div>", unsafe_allow_html=True)

                st.subheader("Scores for all classes")
                score_df = pd.DataFrame({
                    "Class": CLASS_NAMES,
                    "Score": probabilities.astype(float),
                }).sort_values("Score", ascending=False)
                score_df["Score (%)"] = (score_df["Score"] * 100).map(lambda x: f"{x:.2f}%")
                st.bar_chart(score_df.set_index("Class")["Score"])
                st.dataframe(score_df[["Class", "Score (%)"]], hide_index=True, use_container_width=True)

                st.caption(
                    "Confidence is the model's predicted score, not a guarantee that the "
                    "prediction is correct. Model performance depends on training data and image quality."
                )
    except Exception as exc:
        st.error("The image could not be processed or the model could not make a prediction.")
        st.exception(exc)
else:
    st.info("Upload an image to begin.")
    st.subheader("Classes supported by this model")
    st.write(", ".join(CLASS_NAMES))

with st.expander("How to test the application with all five classes"):
    st.markdown(
        """
        1. Start the app and upload one image from `colon_aca`.
        2. Record the predicted class and confidence score.
        3. Repeat with one image from each of `colon_n`, `lung_aca`, `lung_n`, and `lung_scc`.
        4. Record the true folder label, predicted label, and confidence in your project report.
        5. Do not assume every prediction must be correct; report the observed results honestly.
        """
    )
