import numpy as np
import librosa
from typing import Dict

from multimodal_emotion_detection.utils.logger import get_logger
from multimodal_emotion_detection.utils.config import EMOTIONS
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import pickle
from pathlib import Path
from multimodal_emotion_detection.utils.config import AUDIO_MODEL_DIR

logger = get_logger("predict_audio")

# Model architecture must mirror training
class AudioMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)

# Extract features from waveform (mirror training features)
def _extract_features_from_wave(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    try:
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        # Trim silence
        y, _ = librosa.effects.trim(y)
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        rmse = float(np.mean(librosa.feature.rms(y=y)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        contrast = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
        chroma = float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1).astype(np.float32)  # 20 dims
        feats = np.array([
            zcr, rmse, centroid, bandwidth, rolloff, contrast, chroma,
            *mfcc_mean.tolist(),
        ], dtype=np.float32)
        return feats
    except Exception as e:
        logger.error(f"Audio feature extraction failed: {e}")
        return np.zeros(27, dtype=np.float32)

# Lazy-loaded model artifacts
_AUDIO = {
    "loaded": False,
    "model": None,
    "scaler": None,
    "meta": None,
    "device": None,
}

# Attempt to load trained audio model artifacts

def _ensure_audio_model_loaded() -> bool:
    if _AUDIO["loaded"]:
        return True
    try:
        model_dir = AUDIO_MODEL_DIR
        model_path = model_dir / "model.pt"
        scaler_path = model_dir / "scaler.pkl"
        meta_path = model_dir / "meta.json"
        # Ensure files exist
        if not (model_path.exists() and scaler_path.exists() and meta_path.exists()):
            return False
        with open(meta_path, "r") as f:
            meta = json.load(f)
        input_dim = int(meta.get("input_dim", 27))
        num_classes = len(EMOTIONS)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AudioMLP(input_dim=input_dim, num_classes=num_classes).to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        _AUDIO.update({
            "loaded": True,
            "model": model,
            "scaler": scaler,
            "meta": meta,
            "device": device,
        })
        logger.info("Loaded trained audio model artifacts for inference.")
        return True
    except Exception as e:
        logger.warning(f"Failed to load audio model artifacts, using heuristic: {e}")
        _AUDIO["loaded"] = False
        return False

# Heuristic fallback (previous implementation)
def _predict_audio_probs_heuristic(y: np.ndarray, sr: int) -> Dict[str, float]:
    # Feature extraction
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    energy = float(np.mean(y ** 2))
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    pitch = float(np.mean(pitches[pitches > 0])) if np.any(pitches > 0) else 0.0
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = float(np.mean(mfcc))

    # Heuristic mapping
    probs = {e: 0.0 for e in EMOTIONS}
    probs["angry"] = min(1.0, 0.4 * zcr + 0.4 * energy)
    probs["happy"] = min(1.0, 0.5 * pitch + 0.2 * mfcc_mean)
    probs["sad"] = min(1.0, 0.3 * (1 - pitch) + 0.3 * (1 - zcr))
    probs["fear"] = min(1.0, 0.3 * zcr + 0.3 * (1 - pitch))
    probs["surprise"] = min(1.0, 0.2 * pitch + 0.2 * zcr)
    probs["disgust"] = min(1.0, 0.2 * energy + 0.1 * (1 - mfcc_mean))
    probs["neutral"] = max(0.0, 1.0 - sum(probs.values()))

    total = sum(probs.values()) or 1.0
    probs = {k: v / total for k, v in probs.items()}
    logger.info(f"Audio modality (heuristic): {probs}")
    return probs

# Public: predict calibrated probabilities using trained model when available
def predict_audio_probs(y: np.ndarray, sr: int) -> Dict[str, float]:
    if _ensure_audio_model_loaded():
        feats = _extract_features_from_wave(y, sr)
        try:
            scaler = _AUDIO["scaler"]
            X = scaler.transform(feats.reshape(1, -1))
        except Exception:
            # If scaler fails, fallback to raw features
            X = feats.reshape(1, -1)
        device = _AUDIO["device"]
        xb_t = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = _AUDIO["model"](xb_t)
        # Optional temperature calibration
        temp = float(_AUDIO["meta"].get("temperature", 1.0))
        logits = logits / max(1e-6, temp)
        probs_arr = F.softmax(logits, dim=-1).cpu().numpy().reshape(-1)
        # Map to emotions order
        probs = {EMOTIONS[i]: float(probs_arr[i]) for i in range(len(EMOTIONS))}
        logger.info(f"Audio modality (trained): {probs}")
        return probs
    else:
        return _predict_audio_probs_heuristic(y, sr)

# Confidence remains unchanged
def audio_confidence_from_probs(probs: Dict[str, float]) -> float:
    arr = np.array([probs.get(e, 0.0) for e in EMOTIONS], dtype=np.float32)
    s = float(arr.sum()) or 1.0
    arr = arr / s
    entropy = float(-np.sum(arr * np.log(arr + 1e-12)) / np.log(len(EMOTIONS)))
    maxp = float(arr.max())
    return max(0.0, min(1.0, 0.5 * (1.0 - entropy) + 0.5 * maxp))