import cv2
import numpy as np
import soundfile as sf
from typing import Optional, Dict

from multimodal_emotion_detection.utils.logger import get_logger
# Avoid importing image predictor at module load to prevent heavy deps unless needed
from multimodal_emotion_detection.inference.predict_audio import predict_audio_probs
from multimodal_emotion_detection.inference.predict_text import predict_text_probs
from multimodal_emotion_detection.inference.fusion import fuse_probs

logger = get_logger("inference")


def _load_image(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    return img


def _load_audio(path: str):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = y.astype(np.float32)
    return y, sr


def predict_emotion(image_path: Optional[str], audio_path: Optional[str], text_input: Optional[str]) -> Dict:
    modality_probs = {}
    if image_path:
        # Lazy import to avoid DeepFace/TensorFlow init if image modality not used
        from multimodal_emotion_detection.inference.predict_image import predict_image_probs
        img = _load_image(image_path)
        modality_probs["image"] = predict_image_probs(img)
    if audio_path:
        y, sr = _load_audio(audio_path)
        modality_probs["audio"] = predict_audio_probs(y, sr)
    if text_input:
        modality_probs["text"] = predict_text_probs(text_input)

    res = fuse_probs(modality_probs)
    logger.info(f"Predicted: {res['emotion']} | confidence={res['confidence']:.3f}")
    return res