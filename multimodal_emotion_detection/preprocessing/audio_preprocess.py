import pandas as pd
import numpy as np
import librosa
from pathlib import Path
import sys

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger
from utils.config import DATA_DIR

logger = get_logger("audio_preprocess")

# Feature extraction consistent with training

def extract_features(wav_path: Path, sr: int = 16000) -> np.ndarray:
    try:
        y, sr = librosa.load(str(wav_path), sr=sr, mono=True)
        y, _ = librosa.effects.trim(y)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rmse = np.mean(librosa.feature.rms(y=y))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)
        feats = np.array([
            zcr, rmse, centroid, bandwidth, rolloff, contrast, chroma,
            *mfcc_mean.tolist(),
        ], dtype=np.float32)
        return feats
    except Exception as e:
        logger.error(f"Feature extraction failed for {wav_path}: {e}")
        return np.zeros(27, dtype=np.float32)


def build_features(labels_csv: Path, audio_root: Path, out_csv: Path, sr: int = 16000):
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
    df = pd.read_csv(labels_csv)
    if not {"filepath", "label"}.issubset(df.columns):
        raise ValueError("labels.csv must contain columns: filepath,label")

    rows = []
    for _, row in df.iterrows():
        rel = str(row["filepath"])  # relative path under audio_root
        wav_path = audio_root / rel
        if not wav_path.exists():
            logger.warning(f"Missing audio file: {wav_path}")
            continue
        # compute features
        feats = extract_features(wav_path, sr=sr)
        # additional metadata
        try:
            y, sr_loaded = librosa.load(str(wav_path), sr=sr, mono=True)
            y, _ = librosa.effects.trim(y)
            duration = float(librosa.get_duration(y=y, sr=sr_loaded))
            flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
            noise_level = flatness  # simple proxy: higher flatness ~ more noise-like
        except Exception:
            duration = 0.0
            noise_level = 0.0
        feat_dict = {f"feat_{i}": float(feats[i]) for i in range(len(feats))}
        rec = {"filepath": rel, **feat_dict, "duration": duration, "noise_level": noise_level}
        if "speaker_id" in df.columns:
            rec["speaker_id"] = str(row["speaker_id"])
        rows.append(rec)

    if not rows:
        raise RuntimeError("No features computed. Ensure labels.csv and audio files are present.")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    logger.info(f"Wrote features to {out_csv} with shape {out_df.shape}")


def main():
    audio_proc_dir = DATA_DIR / "audio" / "processed"
    labels_csv = audio_proc_dir / "labels.csv"
    out_csv = audio_proc_dir / "features.csv"
    build_features(labels_csv, audio_proc_dir, out_csv)


if __name__ == "__main__":
    main()