from typing import Dict
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

from multimodal_emotion_detection.utils.logger import get_logger
from multimodal_emotion_detection.utils.config import EMOTIONS, TEXT_MODEL_ID, TEXT_MODEL_DIR, TEXT_ALLOW_DOWNLOAD

logger = get_logger("predict_text")

_tokenizer = None
_model = None


def _lazy_load_model():
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return

    # Prefer local pre-baked model directory
    local_dir = str(TEXT_MODEL_DIR)
    local_exists = os.path.isdir(local_dir) and os.path.exists(os.path.join(local_dir, "config.json"))

    if local_exists:
        logger.info(f"Loading text model from local directory: {local_dir}")
        _tokenizer = AutoTokenizer.from_pretrained(local_dir)
        _model = AutoModelForSequenceClassification.from_pretrained(local_dir)
        return

    if not TEXT_ALLOW_DOWNLOAD:
        raise RuntimeError(
            f"Text model not found at {local_dir} and downloads are disabled. "
            f"Please pre-bake the model by running the prebake script or set MME_TEXT_ALLOW_DOWNLOAD=1."
        )

    # Download from HuggingFace and persist to local directory
    logger.info(f"Downloading text model '{TEXT_MODEL_ID}' and caching to {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    _tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
    _model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_ID)
    try:
        _tokenizer.save_pretrained(local_dir)
        _model.save_pretrained(local_dir)
        logger.info("Saved text model/tokenizer to local cache.")
    except Exception as e:
        logger.warning(f"Failed to persist model locally: {e}")


def predict_text_probs(text: str) -> Dict[str, float]:
    if not text or not text.strip():
        return {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}

    _lazy_load_model()
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = _model(**inputs)
    scores = outputs.logits.detach().numpy().flatten().tolist()
    p_pos = float(np.exp(scores[2]) / np.sum(np.exp(scores)))
    p_neg = float(np.exp(scores[0]) / np.sum(np.exp(scores)))

    probs = {e: 0.0 for e in EMOTIONS}
    probs["happy"] = p_pos
    probs["angry"] = 0.5 * p_neg
    probs["sad"] = 0.5 * p_neg
    probs["neutral"] = max(0.0, 1.0 - (probs["happy"] + probs["angry"] + probs["sad"]))

    total = sum(probs.values()) or 1.0
    probs = {k: v / total for k, v in probs.items()}
    logger.info(f"Text modality: {probs}")
    return probs


def text_confidence_from_probs(probs: Dict[str, float]) -> float:
    arr = np.array([probs.get(e, 0.0) for e in EMOTIONS], dtype=np.float32)
    s = float(arr.sum()) or 1.0
    arr = arr / s
    entropy = float(-np.sum(arr * np.log(arr + 1e-12)) / np.log(len(EMOTIONS)))
    maxp = float(arr.max())
    return max(0.0, min(1.0, 0.5 * (1.0 - entropy) + 0.5 * maxp))