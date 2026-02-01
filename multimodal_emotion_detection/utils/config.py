from pathlib import Path
import os

# Centralized config for paths and default weights

# Allow overriding base/data/models directories via environment variables
BASE_DIR = Path(os.getenv("MME_BASE_DIR", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.getenv("MME_DATA_DIR", BASE_DIR / "data"))
MODELS_DIR = Path(os.getenv("MME_MODELS_DIR", BASE_DIR / "models"))

# Optional per-modality model directories (default under MODELS_DIR)
AUDIO_MODEL_DIR = Path(os.getenv("MME_AUDIO_MODEL_DIR", MODELS_DIR / "audio_model"))
IMAGE_MODEL_DIR = Path(os.getenv("MME_IMAGE_MODEL_DIR", MODELS_DIR / "image_model"))
FUSION_MODEL_DIR = Path(os.getenv("MME_FUSION_MODEL_DIR", MODELS_DIR / "fusion_model"))

# HuggingFace text model configuration
TEXT_MODEL_ID = os.getenv("MME_TEXT_MODEL_ID", "cardiffnlp/twitter-roberta-base-sentiment")
TEXT_MODEL_DIR = Path(os.getenv("MME_TEXT_MODEL_DIR", MODELS_DIR / "text_hf"))
# Whether to allow internet downloads at runtime (default: disabled)
TEXT_ALLOW_DOWNLOAD = os.getenv("MME_TEXT_ALLOW_DOWNLOAD", "0") in {"1", "true", "True"}

EMOTIONS = ["happy", "sad", "angry", "neutral", "fear", "surprise", "disgust"]

# Prefer local trained model and ignore unreliable fallbacks
FUSION_WEIGHTS = {
    "image": 0.8,
    "audio": 1.0,
    "text": 1.2,
}