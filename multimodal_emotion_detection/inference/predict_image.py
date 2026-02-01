import numpy as np
# from deepface import DeepFace  # moved to lazy import inside function
from typing import Dict, Optional

from multimodal_emotion_detection.utils.logger import get_logger
from multimodal_emotion_detection.utils.config import EMOTIONS
from multimodal_emotion_detection.utils.config import IMAGE_MODEL_DIR

logger = get_logger("predict_image")

DEEPFACE_TO_EMOTIONS = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "neutral": "neutral",
    "fear": "fear",
    "surprise": "surprise",
    "disgust": "disgust",
}

# Fallback color mapping based on toy data generator
EMOTION_COLORS_RGB = {
    "happy": (255, 215, 0),      # gold
    "sad": (70, 130, 180),       # steel blue
    "angry": (220, 20, 60),      # crimson
    "neutral": (128, 128, 128),  # gray
    "fear": (138, 43, 226),      # blueviolet
    "surprise": (255, 140, 0),   # dark orange
    "disgust": (34, 139, 34),    # forest green
}

# Cache for local trained image model
_IMAGE = {
    "loaded": False,
    "model": None,
    "label_map": None,
    "meta": None,
    "device": None,
    "transform": None,
}


def _ensure_image_model_loaded() -> bool:
    if _IMAGE["loaded"]:
        return True
    try:
        import json
        from pathlib import Path
        from torchvision import models, transforms
        import torch
        model_dir = IMAGE_MODEL_DIR
        model_path = model_dir / "resnet18.pt"
        label_map_path = model_dir / "label_map.json"
        meta_path = model_dir / "meta.json"
        if not (model_path.exists() and label_map_path.exists() and meta_path.exists()):
            return False
        with open(label_map_path, "r") as f:
            label_map = json.load(f)  # {class_name: idx}
        with open(meta_path, "r") as f:
            meta = json.load(f)
        num_classes = int(len(label_map))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Build architecture and load weights
        model = models.resnet18(weights=None)
        model.fc = __import__("torch.nn").nn.Linear(model.fc.in_features, num_classes)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        model.to(device)
        # Inference transform matching training
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        _IMAGE.update({
            "loaded": True,
            "model": model,
            "label_map": label_map,
            "meta": meta,
            "device": device,
            "transform": tf,
        })
        logger.info("Loaded trained image model artifacts for inference.")
        return True
    except Exception as e:
        logger.warning(f"Failed to load image model artifacts, falling back: {e}")
        _IMAGE["loaded"] = False
        return False


def _fallback_color_probs(img: np.ndarray) -> Dict[str, float]:
    # img expected BGR (cv2.imread); convert to RGB for comparison
    if img is None or img.size == 0:
        return {e: 0.0 for e in EMOTIONS}
    rgb = img[..., ::-1]
    mean_rgb = rgb.reshape(-1, 3).mean(axis=0)
    # Compute inverse-distance weights to emotion colors
    weights = {}
    for e, color in EMOTION_COLORS_RGB.items():
        c = np.array(color, dtype=np.float32)
        d = float(np.linalg.norm(mean_rgb - c))
        w = 1.0 / (d + 1e-6)
        weights[e] = w
    total = sum(weights.values()) or 1.0
    probs = {e: float(w / total) for e, w in weights.items()}
    # Ensure all EMOTIONS keys exist
    for e in EMOTIONS:
        probs.setdefault(e, 0.0)
    logger.info(f"Image modality (fallback): {probs}")
    return probs


def _predict_image_probs_local(img: np.ndarray) -> Optional[Dict[str, float]]:
    try:
        if not _ensure_image_model_loaded():
            return None
        import torch
        tf = _IMAGE["transform"]
        device = _IMAGE["device"]
        # Convert BGR numpy array to RGB tensor via ToPILImage
        img_rgb = img[..., ::-1]
        xb = tf(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = _IMAGE["model"](xb)
            probs_arr = __import__("torch.nn.functional").nn.functional.softmax(logits, dim=-1).cpu().numpy().reshape(-1)
        # Map idx to label names
        label_map = _IMAGE["label_map"]
        idx_to_label = {int(v): str(k) for k, v in label_map.items()}
        probs = {idx_to_label.get(i, str(i)): float(probs_arr[i]) for i in range(len(probs_arr))}
        # Normalize into our EMOTIONS keys
        out = {e: float(probs.get(e, 0.0)) for e in EMOTIONS}
        logger.info(f"Image modality (local): {out}")
        return out
    except Exception as e:
        logger.error(f"Local image inference failed: {e}")
        return None


def predict_image_probs(img: np.ndarray) -> Optional[Dict[str, float]]:
    # Prefer local trained model when available
    local = _predict_image_probs_local(img)
    if isinstance(local, dict) and local:
        return local
    # Try DeepFace next, fallback to color-based heuristic
    try:
        from deepface import DeepFace  # lazy import
        res = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False)
        if isinstance(res, list):
            res = res[0]
        raw = res.get("emotion", {})
        total = sum(float(v) for v in raw.values()) or 1.0
        probs = {DEEPFACE_TO_EMOTIONS.get(k.lower(), k.lower()): float(v)/total for k, v in raw.items()}
        for e in EMOTIONS:
            probs.setdefault(e, 0.0)
        logger.info(f"Image modality: {probs}")
        return probs
    except ImportError:
        # DeepFace not installed or incompatible, use fallback
        return _fallback_color_probs(img)
    except Exception as e:
        logger.error(f"Image predictor failed, using fallback: {e}")
        return _fallback_color_probs(img)


def image_confidence_from_probs(probs: Dict[str, float]) -> float:
    arr = np.array([probs.get(e, 0.0) for e in EMOTIONS], dtype=np.float32)
    s = float(arr.sum()) or 1.0
    arr = arr / s
    entropy = float(-np.sum(arr * np.log(arr + 1e-12)) / np.log(len(EMOTIONS)))
    maxp = float(arr.max())
    return max(0.0, min(1.0, 0.5 * (1.0 - entropy) + 0.5 * maxp))