# Multimodal Emotion Detection System

from typing import Dict, Tuple

# Final output type: (emotion_label, confidence)
EMOTIONS = ["happy", "sad", "angry", "neutral", "fear", "surprise", "disgust"]

class EmotionResult:
    def __init__(self, label: str, confidence: float, details: Dict = None):
        self.label = label
        self.confidence = max(0.0, min(1.0, confidence))
        self.details = details or {}

    def to_dict(self):
        return {"label": self.label, "confidence": self.confidence, "details": self.details}

# Config knobs for fusion
FUSION_WEIGHTS = {
    "image": 0.4,
    "audio": 0.35,
    "text": 0.25,
}

# Basic labels normalization map
NORMALIZE = {
    "happiness": "happy",
    "joy": "happy",
    "angry": "angry",
    "anger": "angry",
    "sadness": "sad",
    "fearful": "fear",
    "surprised": "surprise",
    "disgusted": "disgust",
    "neutral": "neutral",
}