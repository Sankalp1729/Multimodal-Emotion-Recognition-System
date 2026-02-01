import json
import pickle
from pathlib import Path
from typing import List, Tuple
import sys

# Ensure package root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import librosa

from utils.logger import get_logger
from utils.config import DATA_DIR, MODELS_DIR, EMOTIONS

logger = get_logger("train_audio")

# Feature extraction per sample

def extract_features(wav_path: Path, sr: int = 16000) -> np.ndarray:
    try:
        y, sr = librosa.load(str(wav_path), sr=sr, mono=True)
        # Trim silence
        y, _ = librosa.effects.trim(y)
        # Basic features
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rmse = np.mean(librosa.feature.rms(y=y))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)  # 20 dims
        # Aggregate
        feats = np.array([
            zcr, rmse, centroid, bandwidth, rolloff, contrast, chroma,
            *mfcc_mean.tolist(),
        ], dtype=np.float32)
        return feats
    except Exception as e:
        logger.error(f"Failed to extract features from {wav_path}: {e}")
        return np.zeros(27, dtype=np.float32)  # 7 basic + 20 mfcc


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


# Updated: support precomputed features.csv with metadata

def load_dataset(labels_csv: Path, audio_root: Path, features_csv: Path | None = None) -> Tuple[np.ndarray, np.ndarray, dict]:
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
    df_labels = pd.read_csv(labels_csv)
    if not {"filepath", "label"}.issubset(df_labels.columns):
        raise ValueError("labels.csv must contain columns: filepath,label")

    df_labels["label"] = df_labels["label"].astype(str).str.lower()
    label_to_idx = {e: i for i, e in enumerate(EMOTIONS)}

    # If features.csv provided and exists, join and use features
    if features_csv is None:
        features_csv = audio_root / "features.csv"
    use_precomputed = features_csv.exists()

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    if use_precomputed:
        df_feat = pd.read_csv(features_csv)
        if not {"filepath"}.issubset(df_feat.columns):
            raise ValueError("features.csv must contain at least filepath and feat_* columns")
        # Identify feature columns by prefix
        feat_cols = [c for c in df_feat.columns if c.startswith("feat_")]
        if not feat_cols:
            raise ValueError("features.csv missing feat_* columns")
        # Join on filepath
        df = df_labels.merge(df_feat["filepath"].to_frame().join(df_feat[feat_cols]), on="filepath", how="inner")
        if df.empty:
            raise RuntimeError("No overlap between labels.csv and features.csv filepaths")
        for _, row in df.iterrows():
            label = str(row["label"])
            if label not in label_to_idx:
                logger.warning(f"Label '{label}' not in EMOTIONS, skipping: {row['filepath']}")
                continue
            feats = row[feat_cols].to_numpy(dtype=np.float32)
            X_list.append(feats)
            y_list.append(label_to_idx[label])
    else:
        for _, row in df_labels.iterrows():
            rel_path = str(row["filepath"])  # relative to audio_root
            label = str(row["label"]).lower()
            wav_path = audio_root / rel_path
            if not wav_path.exists():
                logger.warning(f"Audio file missing, skipping: {wav_path}")
                continue
            if label not in label_to_idx:
                logger.warning(f"Label '{label}' not in EMOTIONS, skipping: {wav_path}")
                continue
            feats = extract_features(wav_path)
            X_list.append(feats)
            y_list.append(label_to_idx[label])

    if not X_list:
        raise RuntimeError("No valid audio samples found. Ensure labels.csv/features.csv and audio files exist and labels match EMOTIONS.")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, label_to_idx


def train_model(X: np.ndarray, y: np.ndarray, num_classes: int, epochs: int = 20, lr: float = 1e-3, batch_size: int = 32, speaker_ids: List[str] | None = None):
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optional speaker-independent split if speaker_ids provided
    if speaker_ids is not None and len(speaker_ids) == len(y):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        (train_idx, val_idx), = gss.split(X_scaled, y, groups=speaker_ids)
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
    else:
        # Ensure stratified test set is large enough to include at least one sample per class
        n_classes = int(len(np.unique(y)))
        n_samples = int(len(y))
        test_size_int = max(n_classes, int(np.ceil(0.2 * n_samples)))
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=test_size_int, random_state=42, stratify=y
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioMLP(input_dim=X.shape[1], num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def batch_iter(Xn, yn, bs):
        N = Xn.shape[0]
        idx = np.arange(N)
        np.random.shuffle(idx)
        for i in range(0, N, bs):
            j = idx[i:i+bs]
            yield Xn[j], yn[j]

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in batch_iter(X_train, y_train, batch_size):
            xb_t = torch.tensor(xb, dtype=torch.float32, device=device)
            yb_t = torch.tensor(yb, dtype=torch.long, device=device)
            optimizer.zero_grad()
            logits = model(xb_t)
            loss = criterion(logits, yb_t)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.shape[0]
        train_loss /= X_train.shape[0]

        # Validation
        model.eval()
        with torch.no_grad():
            xb_t = torch.tensor(X_val, dtype=torch.float32, device=device)
            yb_t = torch.tensor(y_val, dtype=torch.long, device=device)
            logits = model(xb_t)
            val_loss = criterion(logits, yb_t).item()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            acc = (preds == y_val).mean()
        logger.info(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={acc:.4f}")

    return model, scaler, float(acc)


def save_artifacts(model: AudioMLP, scaler: StandardScaler, label_to_idx: dict, acc: float, out_dir: Path, input_dim: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Model state
    model_path = out_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    # Scaler
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    # Label map
    with open(out_dir / "label_map.json", "w") as f:
        json.dump(label_to_idx, f)
    # Meta
    meta = {
        "architecture": "AudioMLP(input_dim->128->64->num_classes)",
        "input_dim": int(input_dim),
        "num_params": int(sum(p.numel() for p in model.parameters())),
        "val_acc": acc,
        "emotions": EMOTIONS,
        "temperature": 1.0,  # optional calibration factor
        "feature_columns": [f"feat_{i}" for i in range(input_dim)],
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved audio model artifacts to {out_dir}")


def main():
    audio_proc_dir = DATA_DIR / "audio" / "processed"
    labels_csv = audio_proc_dir / "labels.csv"
    features_csv = audio_proc_dir / "features.csv"

    try:
        X, y, label_to_idx = load_dataset(labels_csv, audio_proc_dir, features_csv)
    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        logger.info("Expected labels.csv with columns filepath,label under data/audio/processed. Filepaths should be relative to that directory. Optional: features.csv with feat_* columns.")
        return

    logger.info(f"Loaded {X.shape[0]} samples, feature_dim={X.shape[1]}, num_classes={len(label_to_idx)}")

    # Optional: speaker IDs for group split
    speaker_ids = None
    try:
        df_labels = pd.read_csv(labels_csv)
        if "speaker_id" in df_labels.columns:
            # Align speaker_ids to X/y order by reloading via features or on-the-fly loop; for simplicity, assume features.csv used
            if features_csv.exists():
                df_feat = pd.read_csv(features_csv)
                feat_cols = [c for c in df_feat.columns if c.startswith("feat_")]
                df = df_labels.merge(df_feat[["filepath"] + feat_cols], on="filepath", how="inner")
                speaker_ids = df["speaker_id"].astype(str).tolist()
    except Exception as e:
        logger.warning(f"Could not attach speaker_ids for group split: {e}")

    model, scaler, acc = train_model(X, y, num_classes=len(label_to_idx), speaker_ids=speaker_ids)

    out_dir = MODELS_DIR / "audio_model"
    save_artifacts(model, scaler, label_to_idx, acc, out_dir, input_dim=X.shape[1])


if __name__ == "__main__":
    main()