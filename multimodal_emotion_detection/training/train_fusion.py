# Learned Fusion Trainer: Logistic Regression over modality distributions + confidences + availability flags
# Inputs: per-modality class probabilities (7 dims each), per-modality confidence (1 dim each), availability flags (1 dim each)
# Output: final emotion class probabilities over EMOTIONS

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root on sys.path for direct script execution
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

# Lazy heavy imports inside functions to avoid TF/DeepFace overhead if not needed

from multimodal_emotion_detection.utils.config import EMOTIONS, DATA_DIR, MODELS_DIR


def _generic_conf(probs: Dict[str, float]) -> float:
    arr = np.array([probs.get(e, 0.0) for e in EMOTIONS], dtype=np.float32)
    s = float(arr.sum()) or 1.0
    arr = arr / s
    entropy = float(-np.sum(arr * np.log(arr + 1e-12)) / np.log(len(EMOTIONS)))
    maxp = float(arr.max())
    return max(0.0, min(1.0, 0.5 * (1.0 - entropy) + 0.5 * maxp))


# Collect samples

def collect_text_samples(limit_per_class: int = 100) -> List[Tuple[Dict[str, float], str]]:
    from multimodal_emotion_detection.inference.predict_text import predict_text_probs, text_confidence_from_probs
    # Expect data/text/dataset.csv with columns: text, label
    text_csv = DATA_DIR / "text" / "dataset.csv"
    if not text_csv.exists():
        return []
    df = pd.read_csv(text_csv)
    df = df[df["label"].isin(EMOTIONS)]
    out = []
    counts = {e: 0 for e in EMOTIONS}
    for _, row in df.iterrows():
        if counts[row["label"]] >= limit_per_class:
            continue
        probs = predict_text_probs(row["text"])  # returns dict
        conf = text_confidence_from_probs(probs)
        probs["__conf__"] = conf
        out.append((probs, row["label"]))
        counts[row["label"]] += 1
    return out


def collect_audio_samples(limit_per_class: int = 100) -> List[Tuple[Dict[str, float], str]]:
    from multimodal_emotion_detection.inference.predict_audio import predict_audio_probs, audio_confidence_from_probs
    import soundfile as sf
    labels_csv = DATA_DIR / "audio" / "processed" / "labels.csv"
    audio_root = DATA_DIR / "audio" / "processed"
    if not labels_csv.exists():
        return []
    df = pd.read_csv(labels_csv)
    df = df[df["label"].isin(EMOTIONS)]
    out = []
    counts = {e: 0 for e in EMOTIONS}
    for _, row in df.iterrows():
        if counts[row["label"]] >= limit_per_class:
            continue
        rel = row["filepath"]
        path = audio_root / rel
        try:
            y, sr = sf.read(str(path))
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            y = y.astype(np.float32)
        except Exception:
            # fallback to librosa
            import librosa
            y, sr = librosa.load(str(path), sr=16000)
        probs = predict_audio_probs(y, sr)  # returns dict
        conf = audio_confidence_from_probs(probs)
        probs["__conf__"] = conf
        out.append((probs, row["label"]))
        counts[row["label"]] += 1
    return out


def collect_image_samples(limit_per_class: int = 100) -> List[Tuple[Dict[str, float], str]]:
    try:
        from multimodal_emotion_detection.inference.predict_image import predict_image_emotion, image_confidence_from_probs
    except Exception:
        return []
    # Expect data/image/processed/<emotion>/*.png
    image_root = DATA_DIR / "image" / "processed"
    if not image_root.exists():
        return []
    out = []
    counts = {e: 0 for e in EMOTIONS}
    for e in EMOTIONS:
        cls_dir = image_root / e
        if not cls_dir.exists():
            continue
        for p in cls_dir.glob("*.png"):
            if counts[e] >= limit_per_class:
                continue
            probs = predict_image_emotion(str(p))
            conf = image_confidence_from_probs(probs)
            probs["__conf__"] = conf
            out.append((probs, e))
            counts[e] += 1
    return out


# Build fusion dataset

def build_fusion_dataset(include_text: bool = True, include_audio: bool = True, include_image: bool = False, limit_per_class: int = 100) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rows: List[List[float]] = []
    labels: List[str] = []

    # Collect per-modality data
    text_samples = collect_text_samples(limit_per_class) if include_text else []
    audio_samples = collect_audio_samples(limit_per_class) if include_audio else []
    image_samples = collect_image_samples(limit_per_class) if include_image else []

    # Align by sampling balanced counts per class per modality independently and concatenating; fusion model can handle missing modalities via flags
    all_samples = []
    for probs, label in text_samples:
        all_samples.append(({"text": probs}, label))
    for probs, label in audio_samples:
        all_samples.append(({"audio": probs}, label))
    for probs, label in image_samples:
        all_samples.append(({"image": probs}, label))

    # Features: for each modality: EMOTIONS probs + confidence + availability flag
    feature_names: List[str] = []
    for m in ["image", "audio", "text"]:
        feature_names += [f"{m}_prob_{e}" for e in EMOTIONS] + [f"{m}_conf", f"{m}_available"]

    for modality_dict, label in all_samples:
        row: List[float] = []
        for m in ["image", "audio", "text"]:
            probs = modality_dict.get(m)
            if probs is None:
                row += [0.0] * len(EMOTIONS) + [0.0, 0.0]
            else:
                row += [float(probs.get(e, 0.0)) for e in EMOTIONS]
                conf = float(probs.get("__conf__", _generic_conf(probs)))
                row += [conf, 1.0]
        rows.append(row)
        labels.append(label)

    X = np.array(rows, dtype=np.float32)
    y = np.array([EMOTIONS.index(lbl) for lbl in labels], dtype=np.int64)
    return X, y, feature_names


# Train logistic regression fusion

def train_logistic_fusion(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Standardize features
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Split with safeguards for small datasets and stratification
    N = Xs.shape[0]
    n_classes = len(set(y.tolist()))
    base_frac = 0.2
    test_count = max(int(np.ceil(N * base_frac)), n_classes)
    # Ensure train set also has at least one sample per class
    if N - test_count < n_classes:
        # try to shrink test_count while keeping >= n_classes
        test_count = max(n_classes, N // 3)

    if test_count >= N or (N - test_count) < n_classes:
        # Fallback: no proper stratified split possible; train on all and report training accuracy
        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            C=1.0,
            penalty="l2",
        )
        clf.fit(Xs, y)
        y_pred = clf.predict(Xs)
        val_acc = float(accuracy_score(y, y_pred))
        return clf, scaler, {"val_accuracy": val_acc, "note": "Used full-dataset training due to small sample size; accuracy is training accuracy."}

    X_train, X_val, y_train, y_val = train_test_split(Xs, y, test_size=test_count, random_state=42, stratify=y)

    # Use solver that supports multinomial
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
        penalty="l2",
        # multi_class parameter is not passed to maintain compatibility with older sklearn
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    val_acc = float(accuracy_score(y_val, y_pred))

    return clf, scaler, {"val_accuracy": val_acc}


def save_artifacts(clf, scaler, meta: Dict[str, float], feature_names: List[str]):
    out_dir = MODELS_DIR / "fusion_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.pkl", "wb") as f:
        import pickle
        pickle.dump(clf, f)
    with open(out_dir / "scaler.pkl", "wb") as f:
        import pickle
        pickle.dump(scaler, f)
    meta_out = {
        "emotions": EMOTIONS,
        "feature_names": feature_names,
        **meta,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    # Export interpretable weights per feature per class
    # For multinomial logistic regression with lbfgs, coef_.shape = (n_classes, n_features)
    weights_path = out_dir / "feature_weights.csv"
    coef = clf.coef_
    df = pd.DataFrame(coef, columns=feature_names)
    df.insert(0, "class", EMOTIONS)
    df.to_csv(weights_path, index=False)


def save_dataset(X: np.ndarray, y: np.ndarray, feature_names: List[str]):
    out_dir = MODELS_DIR / "fusion_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_path = out_dir / "fusion_dataset.csv"
    df = pd.DataFrame(X, columns=feature_names)
    df.insert(0, "label", [EMOTIONS[i] for i in y])
    df.to_csv(ds_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Train learned fusion (logistic regression)")
    parser.add_argument("--save-dataset", action="store_true")
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument("--include-audio", action="store_true")
    parser.add_argument("--include-image", action="store_true")
    parser.add_argument("--limit-per-class", type=int, default=100)
    args = parser.parse_args()

    X, y, feature_names = build_fusion_dataset(
        include_text=args.include_text,
        include_audio=args.include_audio,
        include_image=args.include_image,
        limit_per_class=args.limit_per_class,
    )
    if X.shape[0] == 0:
        print("No data available to train fusion.")
        return

    clf, scaler, meta = train_logistic_fusion(X, y)
    save_artifacts(clf, scaler, meta, feature_names)
    if args.save_dataset:
        save_dataset(X, y, feature_names)
    print(f"Saved fusion model to: {MODELS_DIR / 'fusion_model'}")
    print(f"Validation accuracy: {meta['val_accuracy']:.4f}")


if __name__ == "__main__":
    main()