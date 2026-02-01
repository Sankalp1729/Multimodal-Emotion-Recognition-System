from typing import Dict, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from core import EMOTIONS

# Text sentiment/emotion model
# We map sentiment to basic emotions; for richer mapping replace model accordingly.

_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
_tokenizer = None
_model = None


def _ensure_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        _model.eval()


SENTIMENT_TO_EMOTION = {
    0: "sad",   # negative -> sad
    1: "neutral",  # neutral -> neutral
    2: "happy",  # positive -> happy
}


def text_emotion_probs(text: str) -> Optional[Dict[str, float]]:
    try:
        _ensure_model()
        inputs = _tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        # Map 3-class sentiment to EMOTIONS
        dist = {e: 0.0 for e in EMOTIONS}
        for i, p in enumerate(probs):
            dist[SENTIMENT_TO_EMOTION[i]] += float(p)
        # small prior for neutral
        dist["neutral"] = max(dist["neutral"], 0.1)
        # Normalize
        s = sum(dist.values())
        for e in EMOTIONS:
            dist[e] = dist[e] / (s + 1e-8)
        return dist
    except Exception:
        return None