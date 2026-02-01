#!/usr/bin/env python3
"""
Lightweight CLI evaluator for weighted fusion baseline.

- Loads small labeled samples (audio-driven) and optionally attaches image/text samples by label
- Runs modality inference
- Applies fusion with configurable weight combinations (audio/image/text)
- Reports accuracy for each configuration and prints the best-performing weights

Keep it small and dependency-light. Image is optional (DeepFace/TensorFlow heavy).
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root on sys.path for direct execution
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from multimodal_emotion_detection.utils.config import EMOTIONS, DATA_DIR

# Lazy imports for modality-specific inference

def _predict_audio_for_file(wav_path: Path) -> Optional[Dict[str, float]]:
    try:
        import soundfile as sf
        y, sr = sf.read(str(wav_path))
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        y = y.astype(np.float32)
    except Exception:
        try:
            import librosa
            y, sr = librosa.load(str(wav_path), sr=16000)
        except Exception:
            return None
    try:
        from multimodal_emotion_detection.inference.predict_audio import predict_audio_probs
        return predict_audio_probs(y, sr)
    except Exception:
        return None


def _predict_image_for_label(label: str, image_root: Path) -> Optional[Dict[str, float]]:
    try:
        from multimodal_emotion_detection.inference.predict_image import predict_image_probs
        from PIL import Image
    except Exception:
        return None
    cls_dir = image_root / label
    if not cls_dir.exists():
        return None
    # Pick first .png deterministically
    files = sorted(cls_dir.glob("*.png"))
    if not files:
        return None
    try:
        img = Image.open(files[0]).convert("RGB")
        img_np = np.array(img)
        return predict_image_probs(img_np)
    except Exception:
        return None


def _predict_image_for_file(img_path: Path) -> Optional[Dict[str, float]]:
    try:
        from multimodal_emotion_detection.inference.predict_image import predict_image_probs
    except Exception:
        return None
    # Try OpenCV (BGR)
    try:
        import cv2
        img = cv2.imread(str(img_path))
        if img is not None:
            return predict_image_probs(img)
    except Exception:
        pass
    # Fallback to PIL (RGB)
    try:
        from PIL import Image
        import numpy as np
        img_rgb = Image.open(img_path).convert("RGB")
        return predict_image_probs(np.array(img_rgb))
    except Exception:
        return None


def _predict_text_for_label(label: str, text_df: pd.DataFrame) -> Optional[Dict[str, float]]:
    try:
        from multimodal_emotion_detection.inference.predict_text import predict_text_probs
    except Exception:
        return None
    # Pick first example deterministically
    candidates = text_df[text_df["label"] == label]
    if candidates.empty:
        return None
    text = str(candidates.iloc[0]["text"])
    try:
        return predict_text_probs(text)
    except Exception:
        return None


# Simple weighted fusion (no rule-based fallbacks)

def fuse_weighted(modality_probs: Dict[str, Dict[str, float]], weights: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
    active = {m: p for m, p in modality_probs.items() if p}
    if not active:
        return "neutral", {e: 0.0 for e in EMOTIONS}
    used_modalities = list(active.keys())
    w = np.array([float(weights.get(m, 0.0)) for m in used_modalities], dtype=np.float32)
    if float(w.sum()) <= 0.0:
        # default to uniform if all zero
        w = np.ones(len(used_modalities), dtype=np.float32)
    w = w / float(w.sum())
    stacked = []
    for m in used_modalities:
        probs = active[m]
        vec = np.array([float(probs.get(e, 0.0)) for e in EMOTIONS], dtype=np.float32)
        stacked.append(vec)
    stacked = np.stack(stacked, axis=0)
    fused = (w[:, None] * stacked).sum(axis=0)
    total = float(fused.sum()) or 1.0
    fused = fused / total
    idx = int(np.argmax(fused))
    return EMOTIONS[idx], {e: float(v) for e, v in zip(EMOTIONS, fused)}


def _parse_range(arg: Optional[str], default_list: List[float]) -> List[float]:
    if not arg:
        return default_list
    s = arg.strip()
    # Support "start:end:step" or comma-separated list
    if ":" in s:
        try:
            parts = s.split(":")
            start = float(parts[0])
            end = float(parts[1])
            step = float(parts[2]) if len(parts) > 2 else 0.1
            vals = []
            x = start
            # include end approximately
            while (step > 0 and x <= end + 1e-9) or (step < 0 and x >= end - 1e-9):
                vals.append(round(x, 6))
                x += step
            return vals
        except Exception:
            pass
    try:
        return [float(x) for x in s.split(",") if x]
    except Exception:
        return default_list


def build_samples(include_audio: bool, include_image: bool, include_text: bool, limit_per_class: int,
                  audio_labels_csv: Path, image_root: Path, text_csv: Path) -> List[Tuple[Dict[str, Dict[str, float]], str]]:
    samples: List[Tuple[Dict[str, Dict[str, float]], str]] = []
    # Load audio labels (driver for sample ids)
    audio_root = audio_labels_csv.parent
    if include_audio and audio_labels_csv.exists():
        df_audio = __import__("pandas").read_csv(audio_labels_csv)
        df_audio = df_audio[df_audio["label"].isin(EMOTIONS)]
    else:
        df_audio = __import__("pandas").DataFrame(columns=["filepath", "label"])  # empty

    # Load text dataset for label-matched selection
    if include_text and text_csv.exists():
        df_text = __import__("pandas").read_csv(text_csv)
        df_text = df_text[df_text["label"].isin(EMOTIONS)]
    else:
        df_text = __import__("pandas").DataFrame(columns=["text", "label"])  # empty

    # Count per class to limit
    counts = {e: 0 for e in EMOTIONS}

    # Fallback: if no audio labels to drive iteration, allow image-driven sampling
    if (not include_audio) or df_audio.empty:
        if include_image and Path(image_root).exists():
            for label in EMOTIONS:
                cls_dir = Path(image_root) / label
                if not cls_dir.exists():
                    continue
                for p in sorted(cls_dir.glob("*.png")):
                    if counts[label] >= limit_per_class:
                        break
                    ip = _predict_image_for_file(p)
                    if ip:
                        samples.append(({"image": ip}, label))
                        counts[label] += 1
        return samples

    # Iterate audio-driven rows; if audio not included, we still iterate to have labels
    rows_iter = df_audio.iterrows() if not df_audio.empty else []
    for _, row in rows_iter:
        label = str(row["label"]) if "label" in row else None
        if label not in EMOTIONS:
            continue
        if counts[label] >= limit_per_class:
            continue
        sample_modalities: Dict[str, Dict[str, float]] = {}
        # Audio modality
        if include_audio:
            rel = str(row["filepath"]) if "filepath" in row else None
            wav_path = (audio_root / rel) if rel else None
            if wav_path and wav_path.exists():
                ap = _predict_audio_for_file(wav_path)
                if ap:
                    sample_modalities["audio"] = ap
        # Image modality: pick any image under label
        if include_image:
            ip = _predict_image_for_label(label, image_root)
            if ip:
                sample_modalities["image"] = ip
        # Text modality: pick any text under label
        if include_text:
            tp = _predict_text_for_label(label, df_text)
            if tp:
                sample_modalities["text"] = tp
        # Only consider sample if at least one modality present
        if sample_modalities:
            samples.append((sample_modalities, label))
            counts[label] += 1
    return samples


def evaluate(samples: List[Tuple[Dict[str, Dict[str, float]], str]], audio_ws: List[float], image_ws: List[float], text_ws: List[float],
             verbose: bool = False, per_class: bool = False) -> Dict[str, object]:
    if not samples:
        return {"best": None, "results": [], "note": "No samples to evaluate."}
    results = []
    best = None
    best_acc = -1.0
    for wa in audio_ws:
        for wi in image_ws:
            for wt in text_ws:
                weights = {"audio": wa, "image": wi, "text": wt}
                total = 0
                correct = 0
                # per-class counters
                cls_total = {e: 0 for e in EMOTIONS}
                cls_correct = {e: 0 for e in EMOTIONS}
                for modalities, label in samples:
                    pred, _ = fuse_weighted(modalities, weights)
                    total += 1
                    cls_total[label] += 1
                    if pred == label:
                        correct += 1
                        cls_correct[label] += 1
                acc = float(correct) / float(total) if total else 0.0
                entry = {
                    "weights": weights,
                    "accuracy": acc,
                }
                if per_class:
                    entry["per_class_accuracy"] = {e: (float(cls_correct[e]) / float(cls_total[e]) if cls_total[e] else 0.0) for e in EMOTIONS}
                results.append(entry)
                if verbose:
                    print(f"Weights {weights} -> accuracy {acc:.4f}")
                if acc > best_acc:
                    best_acc = acc
                    best = entry
    return {"best": best, "results": results}


def main():
    parser = argparse.ArgumentParser(description="CLI evaluator for weighted fusion baseline")
    parser.add_argument("--include-audio", action="store_true", help="Include audio modality")
    parser.add_argument("--include-image", action="store_true", help="Include image modality (DeepFace heavy)")
    parser.add_argument("--include-text", action="store_true", help="Include text modality")
    parser.add_argument("--limit-per-class", type=int, default=10, help="Max samples per class")
    parser.add_argument("--audio-labels", type=str, default=str(DATA_DIR / "audio" / "processed" / "labels.csv"), help="Path to audio labels.csv (filepath,label)")
    parser.add_argument("--image-root", type=str, default=str(DATA_DIR / "image" / "processed"), help="Root of image processed data")
    parser.add_argument("--text-csv", type=str, default=str(DATA_DIR / "text" / "dataset.csv"), help="Path to text dataset.csv (text,label)")
    parser.add_argument("--audio-weights", type=str, default="0.5:0.9:0.1", help="Range or list for audio weights")
    parser.add_argument("--image-weights", type=str, default="0.1:0.5:0.1", help="Range or list for image weights")
    parser.add_argument("--text-weights", type=str, default="0.1,0.2,0.3", help="Range or list for text weights")
    parser.add_argument("--verbose", action="store_true", help="Verbose per-config logging")
    parser.add_argument("--per-class", action="store_true", help="Report per-class accuracy")

    args = parser.parse_args()

    include_audio = bool(args.include_audio)
    include_image = bool(args.include_image)
    include_text = bool(args.include_text)

    audio_ws = _parse_range(args.audio_weights, [0.5, 0.6, 0.7, 0.8, 0.9])
    image_ws = _parse_range(args.image_weights, [0.1, 0.2, 0.3, 0.4, 0.5])
    text_ws = _parse_range(args.text_weights, [0.1, 0.2, 0.3])

    samples = build_samples(
        include_audio=include_audio,
        include_image=include_image,
        include_text=include_text,
        limit_per_class=int(args.limit_per_class),
        audio_labels_csv=Path(args.audio_labels),
        image_root=Path(args.image_root),
        text_csv=Path(args.text_csv),
    )

    if not samples:
        print("No samples found. Ensure toy data is generated: notebooks/generate_toy_data.py")
        sys.exit(1)

    res = evaluate(samples, audio_ws=audio_ws, image_ws=image_ws, text_ws=text_ws, verbose=bool(args.verbose), per_class=bool(args.per_class))

    best = res.get("best")
    print("\n=== Weighted Fusion Baseline Results ===")
    if best:
        print(f"Best weights: {best['weights']} -> Accuracy: {best['accuracy']:.4f}")
        if args.per_class and "per_class_accuracy" in best:
            print("Per-class accuracy (best):")
            for e in EMOTIONS:
                v = best["per_class_accuracy"].get(e, 0.0)
                print(f"  {e}: {v:.4f}")
    else:
        print("No valid evaluation results.")

    # Optional: save results to CSV for reference
    out_dir = PROJECT_ROOT / "multimodal_emotion_detection" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "evaluate_fusion_results.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["audio_weight", "image_weight", "text_weight", "accuracy"]
        w.writerow(header)
        for entry in res.get("results", []):
            w.writerow([
                entry["weights"].get("audio", 0.0),
                entry["weights"].get("image", 0.0),
                entry["weights"].get("text", 0.0),
                entry.get("accuracy", 0.0),
            ])
    print(f"Saved sweep results to {out_path}")


if __name__ == "__main__":
    main()