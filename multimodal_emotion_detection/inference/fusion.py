import numpy as np
from typing import Dict, List
import pickle

from multimodal_emotion_detection.utils.config import EMOTIONS, FUSION_WEIGHTS, MODELS_DIR
from multimodal_emotion_detection.utils.config import FUSION_MODEL_DIR

# Cache for learned fusion model
_FUSION = {
    "loaded": False,
    "clf": None,
    "scaler": None,
    "meta": None,
}


def _ensure_fusion_model_loaded() -> bool:
    if _FUSION["loaded"]:
        return True
    try:
-        model_dir = MODELS_DIR / "fusion_model"
+        model_dir = FUSION_MODEL_DIR
        model_path = model_dir / "model.pkl"
        scaler_path = model_dir / "scaler.pkl"
        meta_path = model_dir / "meta.json"
        if not (model_path.exists() and meta_path.exists()):
            return False
        with open(meta_path, "r") as f:
            meta = __import__("json").load(f)
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
        scaler = None
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        _FUSION.update({
            "loaded": True,
            "clf": clf,
            "scaler": scaler,
            "meta": meta,
        })
        return True
    except Exception:
        _FUSION["loaded"] = False
        return False


# Build feature vector consistent with training script, with lazy confidence imports
_MODALITIES = ["image", "audio", "text"]


def _generic_conf(probs: Dict[str, float]) -> float:
    arr = np.array([probs.get(e, 0.0) for e in EMOTIONS], dtype=np.float32)
    s = float(arr.sum()) or 1.0
    arr = arr / s
    entropy = float(-np.sum(arr * np.log(arr + 1e-12)) / np.log(len(EMOTIONS)))
    maxp = float(arr.max())
    return max(0.0, min(1.0, 0.5 * (1.0 - entropy) + 0.5 * maxp))


def _features_for_learned(modality_probs: Dict[str, Dict[str, float]]) -> np.ndarray:
    feats: List[float] = []
    for m in _MODALITIES:
        probs = modality_probs.get(m)
        if probs:
            feats += [float(probs.get(e, 0.0)) for e in EMOTIONS]
            # Lazy import per modality confidence; fallback to generic
            try:
                if m == "image":
                    from multimodal_emotion_detection.inference.predict_image import image_confidence_from_probs
                    conf = float(image_confidence_from_probs(probs))
                elif m == "audio":
                    from multimodal_emotion_detection.inference.predict_audio import audio_confidence_from_probs
                    conf = float(audio_confidence_from_probs(probs))
                else:
                    from multimodal_emotion_detection.inference.predict_text import text_confidence_from_probs
                    conf = float(text_confidence_from_probs(probs))
            except Exception:
                conf = _generic_conf(probs)
            feats += [conf, 1.0]
        else:
            feats += [0.0] * len(EMOTIONS) + [0.0, 0.0]
    return np.array(feats, dtype=np.float32)


# Late fusion via learned logistic regression when available; else weighted probability averaging

def fuse_probs(modality_probs: Dict[str, Dict[str, float]]) -> Dict:
    active = {m: p for m, p in modality_probs.items() if p}
    if not active:
        details = {"reason": "no modalities"}
        return {"emotion": "neutral", "confidence": 0.0, "probs": {e: 0.0 for e in EMOTIONS}, "details": details}

    # Try learned fusion first
    if _ensure_fusion_model_loaded():
        try:
            x = _features_for_learned(modality_probs)
            scaler = _FUSION.get("scaler")
            if scaler is not None:
                x = scaler.transform(x.reshape(1, -1))
            else:
                x = x.reshape(1, -1)
            clf = _FUSION["clf"]
            probs_arr = clf.predict_proba(x)[0]
            # Map to emotions (assuming meta['emotions'] order). If mismatch, fallback to config EMOTIONS order
            meta_emotions = _FUSION.get("meta", {}).get("emotions", EMOTIONS)
            dist = {e: float(p) for e, p in zip(meta_emotions, probs_arr)}
            # Reorder to current EMOTIONS list
            fused_dist = {e: float(dist.get(e, 0.0)) for e in EMOTIONS}
            emotion = max(fused_dist, key=fused_dist.get)
            confidence = float(fused_dist[emotion])
            details = {
                "used_learned_fusion": True,
                "per_modality": active,
                "feature_names": _FUSION.get("meta", {}).get("feature_names", []),
                "feature_vector": {fn: float(v) for fn, v in zip(_FUSION.get("meta", {}).get("feature_names", []), _features_for_learned(modality_probs).tolist())},
                "fused_distribution": fused_dist,
                "val_accuracy": _FUSION.get("meta", {}).get("val_accuracy"),
            }
            return {"emotion": emotion, "confidence": confidence, "probs": fused_dist, "details": details}
        except Exception:
            # Fall through to weighted fusion
            pass

    # Baseline: weighted-confidence averaging
    def _conf(probs: Dict[str, float]) -> float:
        arr = np.array([probs.get(e, 0.0) for e in EMOTIONS], dtype=np.float32)
        s = float(arr.sum()) or 1.0
        arr = arr / s
        entropy = float(-np.sum(arr * np.log(arr + 1e-12)) / np.log(len(EMOTIONS)))
        maxp = float(arr.max())
        # Combine entropy-based sharpness and max probability
        return max(0.0, min(1.0, 0.5 * (1.0 - entropy) + 0.5 * maxp))

    base = np.array([FUSION_WEIGHTS.get(m, 1.0) for m in active.keys()], dtype=np.float32)
    confs = np.array([_conf(p) for p in active.values()], dtype=np.float32)
    weights = base * confs
    if float(weights.sum()) == 0.0:
        weights = base
    weights = weights / (float(weights.sum()) or 1.0)

    stacked = []
    for probs in active.values():
        vec = np.array([probs.get(e, 0.0) for e in EMOTIONS], dtype=np.float32)
        stacked.append(vec)
    stacked = np.stack(stacked, axis=0)

    fused_vec = (weights[:, None] * stacked).sum(axis=0)
    total = float(fused_vec.sum()) or 1.0
    fused_dist = {e: float(v / total) for e, v in zip(EMOTIONS, fused_vec)}
    emotion = max(fused_dist, key=fused_dist.get)
    confidence = float(fused_dist[emotion])

    details = {
        "used_learned_fusion": False,
        "per_modality": active,
        "used_weights": {m: float(w) for m, w in zip(active.keys(), weights)},
        "confidences": {m: float(c) for m, c in zip(active.keys(), confs)},
        "fused_distribution": fused_dist,
    }

    # Rule-based fallbacks (post-fusion, baseline path only)
    MODALITY_CONF_THRESHOLD = 0.25
    FINAL_ENTROPY_THRESHOLD = 0.85
    FINAL_CONF_THRESHOLD = 0.4

    # Ignore unreliable modalities by confidence
    ignored = [m for m, c in zip(list(active.keys()), list(confs)) if c < MODALITY_CONF_THRESHOLD]
    if ignored and len(ignored) < len(active):
        reliable = [(m, active[m], c) for m, c in zip(list(active.keys()), list(confs)) if m not in ignored]
        base_rel = np.array([FUSION_WEIGHTS.get(m, 1.0) for (m, _, _) in reliable], dtype=np.float32)
        confs_rel = np.array([c for (_, _, c) in reliable], dtype=np.float32)
        weights_rel = base_rel * confs_rel
        if float(weights_rel.sum()) == 0.0:
            weights_rel = base_rel
        weights_rel = weights_rel / (float(weights_rel.sum()) or 1.0)
        stacked_rel = np.stack([np.array([probs.get(e, 0.0) for e in EMOTIONS], dtype=np.float32) for (_, probs, _) in reliable], axis=0)
        fused_rel = (weights_rel[:, None] * stacked_rel).sum(axis=0)
        total_rel = float(fused_rel.sum()) or 1.0
        fused_rel = fused_rel / total_rel
        fused_dist = {e: float(v) for e, v in zip(EMOTIONS, fused_rel)}
        details["fallback_applied"] = True
        details["ignored_modalities"] = ignored
        details["used_weights"] = {m: float(w) for m, w in zip([m for (m, _, _) in reliable], weights_rel)}
        details["confidences"] = {m: float(c) for m, c in zip([m for (m, _, _) in reliable], confs_rel)}
        details["fused_distribution"] = fused_dist

    # Entropy-based rejection of final fused distribution
    fused_vec_final = np.array([details["fused_distribution"][e] for e in EMOTIONS], dtype=np.float32)
    entropy = float(-np.sum(fused_vec_final * np.log(fused_vec_final + 1e-12)) / np.log(len(EMOTIONS)))
    maxp = float(fused_vec_final.max())
    if entropy > FINAL_ENTROPY_THRESHOLD and maxp < FINAL_CONF_THRESHOLD:
        details["rejection"] = {
            "type": "high_entropy_low_conf",
            "entropy": float(entropy),
            "maxp": float(maxp),
            "entropy_threshold": float(FINAL_ENTROPY_THRESHOLD),
            "conf_threshold": float(FINAL_CONF_THRESHOLD),
        }
        emotion = "neutral"
        confidence = 0.0

    return {"emotion": emotion, "confidence": confidence, "probs": fused_dist, "details": details}