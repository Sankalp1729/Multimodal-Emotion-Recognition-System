import numpy as np
import librosa
from typing import Dict, Optional
from core import EMOTIONS

# Simple audio emotion estimator using spectral/voice features and a heuristic classifier
# For production, replace with a trained SER (speech emotion recognition) model.


def audio_emotion_probs(y: np.ndarray, sr: int) -> Optional[Dict[str, float]]:
    try:
        # Extract features
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        energy = np.mean(librosa.feature.rms(y))
        pitch = np.mean(librosa.yin(y, fmin=50, fmax=400, sr=sr))
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

        # Heuristic mapping: energetic & high pitch -> happy/surprise; low energy -> sad; high zcr -> angry
        scores = {e: 0.0 for e in EMOTIONS}
        scores["angry"] = float(zcr * 2 + (pitch/300))
        scores["happy"] = float(energy * 3 + (pitch/300))
        scores["sad"] = float(max(0.0, 0.8 - energy))
        scores["neutral"] = 0.5
        scores["fear"] = float(max(0.0, (0.7 - energy) + (0.6 - pitch/300)))
        scores["surprise"] = float((pitch/300) + zcr)
        scores["disgust"] = float(zcr * 0.5)

        # Softmax normalization
        vec = np.array([scores[e] for e in EMOTIONS])
        vec = np.exp(vec - np.max(vec))
        probs = vec / (np.sum(vec) + 1e-8)
        return {e: float(p) for e, p in zip(EMOTIONS, probs)}
    except Exception:
        return None