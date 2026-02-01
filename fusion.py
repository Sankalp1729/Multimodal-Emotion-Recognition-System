import numpy as np
from typing import Dict
from core import EMOTIONS, FUSION_WEIGHTS, NORMALIZE, EmotionResult

# Late fusion via weighted probability averaging
# Inputs: per-modality probability distributions over EMOTIONS
# Missing modalities handled by renormalizing used weights

def normalize_label(label: str) -> str:
    return NORMALIZE.get(label.lower(), label.lower())

# Confidence measure for a probability distribution (entropy + max probability)
def _conf(probs: Dict[str, float]) -> float:
    arr = np.array([probs.get(e, 0.0) for e in EMOTIONS], dtype=np.float32)
    s = float(arr.sum()) or 1.0
    arr = arr / s
    entropy = float(-np.sum(arr * np.log(arr + 1e-12)) / np.log(len(EMOTIONS)))
    maxp = float(arr.max())
    return max(0.0, min(1.0, 0.5 * (1.0 - entropy) + 0.5 * maxp))


def fuse_probs(modality_probs: Dict[str, Dict[str, float]]) -> EmotionResult:
    # Filter modalities that provided data
    active_modalities = {m: p for m, p in modality_probs.items() if p}
    if not active_modalities:
        return EmotionResult("neutral", 0.0, {"reason": "no modalities"})

    # Base weights from config
    base = np.array([FUSION_WEIGHTS.get(m, 1.0) for m in active_modalities.keys()], dtype=np.float32)
    # Confidence per modality
    confs = np.array([_conf(p) for p in active_modalities.values()], dtype=np.float32)
    # Confidence-weighted fusion weights
    weights = base * confs
    if float(weights.sum()) == 0.0:
        weights = base
    weights = weights / (float(weights.sum()) or 1.0)

    # Stack probabilities aligned by EMOTIONS
    stacked = []
    for m, probs in active_modalities.items():
        vec = np.array([probs.get(e, 0.0) for e in EMOTIONS], dtype=np.float32)
        stacked.append(vec)
    stacked = np.stack(stacked, axis=0)

    # Weighted average and renormalize
    fused_vec = (weights[:, None] * stacked).sum(axis=0)
    total = float(fused_vec.sum()) or 1.0
    fused_vec = fused_vec / total

    idx = int(np.argmax(fused_vec))
    label = EMOTIONS[idx]
    confidence = float(fused_vec[idx])

    details = {
        "per_modality": active_modalities,
        "used_weights": {m: float(w) for m, w in zip(active_modalities.keys(), weights)},
        "confidences": {m: float(c) for m, c in zip(active_modalities.keys(), confs)},
        "fused_distribution": {e: float(v) for e, v in zip(EMOTIONS, fused_vec)},
    }

    # Rule-based fallbacks (post-fusion)
    MODALITY_CONF_THRESHOLD = 0.25
    FINAL_ENTROPY_THRESHOLD = 0.85
    FINAL_CONF_THRESHOLD = 0.4

    ignored = [m for m, c in zip(active_modalities.keys(), confs) if c < MODALITY_CONF_THRESHOLD]
    if ignored and len(ignored) < len(active_modalities):
        reliable_items = [(m, active_modalities[m], c) for (m, c) in zip(active_modalities.keys(), confs) if m not in ignored]
        base_rel = np.array([FUSION_WEIGHTS.get(m, 1.0) for (m, _, _) in reliable_items], dtype=np.float32)
        confs_rel = np.array([c for (_, _, c) in reliable_items], dtype=np.float32)
        weights_rel = base_rel * confs_rel
        if float(weights_rel.sum()) == 0.0:
            weights_rel = base_rel
        weights_rel = weights_rel / (float(weights_rel.sum()) or 1.0)
        stacked_rel = np.stack([np.array([probs.get(e, 0.0) for e in EMOTIONS], dtype=np.float32) for (_, probs, _) in reliable_items], axis=0)
        fused_vec_rel = (weights_rel[:, None] * stacked_rel).sum(axis=0)
        total_rel = float(fused_vec_rel.sum()) or 1.0
        fused_vec_rel = fused_vec_rel / total_rel
        details["fallback_applied"] = True
        details["ignored_modalities"] = ignored
        details["used_weights"] = {m: float(w) for m, w in zip([m for (m, _, _) in reliable_items], weights_rel)}
        details["confidences"] = {m: float(c) for m, c in zip([m for (m, _, _) in reliable_items], confs_rel)}
        details["fused_distribution"] = {e: float(v) for e, v in zip(EMOTIONS, fused_vec_rel)}

    # Recompute label/confidence based on potentially updated fused distribution
    fused_vec_final = np.array([details["fused_distribution"][e] for e in EMOTIONS], dtype=np.float32)
    idx = int(np.argmax(fused_vec_final))
    label = EMOTIONS[idx]
    confidence = float(fused_vec_final[idx])

    # Entropy-based rejection on final fused distribution
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
        label = "neutral"
        confidence = 0.0

    return EmotionResult(label, confidence, details)