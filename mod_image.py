import numpy as np
from deepface import DeepFace
from typing import Dict, Optional
from core import EMOTIONS, NORMALIZE

# Image Emotion Model using DeepFace analyze
# Input: path or numpy frame (BGR/RGB)
# Output: probabilities over EMOTIONS (mapped from DeepFace labels)

DEEPFACE_TO_EMOTIONS = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "neutral": "neutral",
    "fear": "fear",
    "surprise": "surprise",
    "disgust": "disgust",
}


def image_emotion_probs(img: np.ndarray) -> Optional[Dict[str, float]]:
    try:
        res = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False)
        # DeepFace returns dict or list; normalize
        if isinstance(res, list):
            res = res[0]
        raw = res.get("emotion", {})
        # Normalize to our EMOTIONS order, scale to 0..1
        total = sum(float(v) for v in raw.values()) or 1.0
        probs = {DEEPFACE_TO_EMOTIONS.get(k.lower(), k.lower()): float(v)/total for k, v in raw.items()}
        # Ensure all EMOTIONS present
        for e in EMOTIONS:
            probs.setdefault(e, 0.0)
        return probs
    except Exception as e:
        return None