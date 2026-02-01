import numpy as np
from typing import Optional, Dict

from multimodal_emotion_detection.utils.logger import get_logger
# Avoid importing heavy deps at module load unless needed
from multimodal_emotion_detection.inference.predict_audio import predict_audio_probs
from multimodal_emotion_detection.inference.predict_text import predict_text_probs
from multimodal_emotion_detection.inference.fusion import fuse_probs

logger = get_logger("inference")


def _load_image(path: str) -> Optional[np.ndarray]:
    try:
        import cv2  # lazy import to avoid CV2 at module import time
        img = cv2.imread(path)
        if img is None:
            return None
        # Try face detection and crop to the largest face to improve emotion inference
        try:
            cascade_path = __import__("os").path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            if __import__("os").path.exists(cascade_path):
                face_cascade = cv2.CascadeClassifier(cascade_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                if len(faces) > 0:
                    # Pick the largest face
                    x, y, w, h = sorted(list(faces), key=lambda b: b[2] * b[3], reverse=True)[0]
                    img = img[y:y+h, x:x+w]
        except Exception as _:
            # If detection fails, just use the original image
            pass
        return img
    except Exception as e:
        logger.warning(f"Failed to load image via cv2: {e}")
        return None


def _load_audio(path: str):
    try:
        import soundfile as sf  # lazy import to avoid libsndfile issues at import time
        y, sr = sf.read(path)
        import numpy as _np
        if y.ndim > 1:
            y = _np.mean(y, axis=1)
        y = _np.asarray(y, dtype=_np.float32)
        return y, sr
    except Exception as e:
        logger.warning(f"Failed to load audio via soundfile: {e}")
        return None, None


def predict_emotion(image_path: Optional[str], audio_path: Optional[str], text_input: Optional[str]) -> Dict:
    modality_probs = {}
    if image_path:
        # Lazy import to avoid DeepFace/TensorFlow init if image modality not used
        from multimodal_emotion_detection.inference.predict_image import predict_image_probs
        img = _load_image(image_path)
        if img is not None:
            modality_probs["image"] = predict_image_probs(img)
        else:
            logger.warning(f"Image at '{image_path}' could not be loaded.")
    if audio_path:
        y, sr = _load_audio(audio_path)
        if y is not None and sr is not None:
            modality_probs["audio"] = predict_audio_probs(y, sr)
        else:
            logger.warning(f"Audio at '{audio_path}' could not be loaded.")
    if text_input:
        modality_probs["text"] = predict_text_probs(text_input)

    res = fuse_probs(modality_probs)
    logger.info(f"Predicted: {res['emotion']} | confidence={res['confidence']:.3f}")
    return res