import os
import tempfile
from pathlib import Path
import json
import sys

import streamlit as st
import pandas as pd

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multimodal_emotion_detection.inference.predict_emotion import predict_emotion
from multimodal_emotion_detection.utils.config import DATA_DIR

st.set_page_config(page_title="Multimodal Emotion Detection", page_icon="🎭", layout="wide")
st.title("🎭 Multimodal Emotion Detection")
st.caption("Upload image/audio, enter text, or pick samples. Click Predict to see fused emotion and probabilities.")

# Helpers
_SUFFIX_MAP = {
    "image": {"png": ".png", "jpg": ".jpg", "jpeg": ".jpg", "webp": ".webp"},
    "audio": {"wav": ".wav"},
}


def _save_upload(tmp_kind: str, upload) -> str:
    if upload is None:
        return None
    suffix = _SUFFIX_MAP[tmp_kind].get(upload.name.split(".")[-1].lower(), "")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.read())
        return tmp.name


def _list_samples():
    audio_clips = sorted((DATA_DIR / "audio" / "processed" / "clips").glob("*.wav"))
    image_files = sorted((DATA_DIR / "image" / "processed").rglob("*.png"))
    return image_files, audio_clips


def _render_result(result: dict):
    if not isinstance(result, dict) or not result:
        st.warning("No result to display.")
        return
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric(label="Emotion", value=result.get("emotion"))
        conf = result.get("confidence", 0.0)
        st.write(f"Confidence: {conf:.3f}")
    with c2:
        probs = result.get("probs", {})
        if isinstance(probs, dict) and probs:
            df = pd.DataFrame({"emotion": list(probs.keys()), "probability": list(probs.values())})
            df = df.set_index("emotion").sort_index()
            st.bar_chart(df)
        else:
            st.info("No probability distribution available.")
    with st.expander("Raw details"):
        st.json(result)


# Tabs for a clearer UI
quick_tab, upload_tab, samples_tab, settings_tab = st.tabs(["Quick Start", "Upload", "Samples", "Settings"])

with quick_tab:
    st.subheader("Try a default sample")
    st.write("We'll auto-pick the first available image and audio from DATA_DIR and use a simple text.")
    image_files, audio_clips = _list_samples()
    if not image_files and not audio_clips:
        st.warning("No sample files found under DATA_DIR. Please go to Upload tab to provide inputs.")
    else:
        if st.button("Run Default Sample", type="primary"):
            img_path = str(image_files[0]) if image_files else None
            aud_path = str(audio_clips[0]) if audio_clips else None
            text_val = "hello world"
            with st.spinner("Running prediction..."):
                try:
                    result = predict_emotion(img_path, aud_path, text_val)
                    _render_result(result)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

with upload_tab:
    st.subheader("Upload your inputs")
    st.write("You can provide any combination of image, audio (WAV), and text.")
    with st.form("upload_form"):
        col_img, col_aud, col_txt = st.columns([1, 1, 1])
        with col_img:
            image_file = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"], help="Upload a face/photo or any image.")
            if image_file is not None:
                st.image(image_file, caption="Image preview", use_column_width=True)
        with col_aud:
            audio_file = st.file_uploader("Audio (WAV)", type=["wav"], help="Upload a short speech/audio clip.")
            if audio_file is not None:
                try:
                    st.audio(audio_file.read(), format="audio/wav")
                except Exception:
                    st.info("Audio selected.")
        with col_txt:
            text_input = st.text_area("Text", placeholder="Type a sentence describing emotion...")
            if text_input:
                st.code(text_input)

        submitted = st.form_submit_button("Predict", type="primary")
        if submitted:
            img_path = _save_upload("image", image_file) if image_file else None
            aud_path = _save_upload("audio", audio_file) if audio_file else None
            text_val = text_input if (text_input and text_input.strip()) else None
            if not any([img_path, aud_path, text_val]):
                st.warning("Please provide at least one modality: image, audio, or text.")
            else:
                with st.spinner("Running prediction..."):
                    try:
                        result = predict_emotion(img_path, aud_path, text_val)
                        _render_result(result)
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

with samples_tab:
    st.subheader("Pick samples from DATA_DIR")
    image_files, audio_clips = _list_samples()
    if not image_files and not audio_clips:
        st.warning("No sample files found in DATA_DIR. Please use the Upload tab.")
    else:
        with st.form("samples_form"):
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                img_choice = st.selectbox(
                    "Sample image",
                    options=["(none)"] + [str(p) for p in image_files],
                    index=0,
                )
            with col2:
                aud_choice = st.selectbox(
                    "Sample audio",
                    options=["(none)"] + [str(p) for p in audio_clips],
                    index=0,
                )
            with col3:
                text_val = st.text_input("Sample text", value="hello world")

            run_samples = st.form_submit_button("Predict with selected samples", type="primary")
            if run_samples:
                img_path = None if img_choice == "(none)" else img_choice
                aud_path = None if aud_choice == "(none)" else aud_choice
                txt = text_val if (text_val and text_val.strip()) else None
                if not any([img_path, aud_path, txt]):
                    st.warning("Please select at least one sample or enter text.")
                else:
                    with st.spinner("Running prediction..."):
                        try:
                            result = predict_emotion(img_path, aud_path, txt)
                            _render_result(result)
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")

with settings_tab:
    st.subheader("Settings")
    st.write("Local inference is used by default. Optionally, you can try the FastAPI server if available.")
    use_api = st.checkbox("Use FastAPI server (experimental)", value=False, help="Calls /predict via JSON. Leave off to run locally.")
    server_url = st.text_input("API base URL", value="http://localhost:8000", help="E.g., http://localhost:8000")

    st.info("If you enable the API option, the Upload/Samples tabs will still display results from local inference in this UI. A future enhancement can compare Local vs API side-by-side.")