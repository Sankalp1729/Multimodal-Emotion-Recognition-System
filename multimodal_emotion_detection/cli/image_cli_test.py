import json
from pathlib import Path
import sys

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multimodal_emotion_detection.inference.predict_emotion import predict_emotion
from multimodal_emotion_detection.utils.config import DATA_DIR


def main():
    img_root = DATA_DIR / "image" / "processed"
    # Collect a few sample images across classes
    paths = sorted(list(img_root.rglob("*.png")))[:3]
    print("Using image files:", [str(p) for p in paths])
    for p in paths:
        res = predict_emotion(str(p), None, None)
        print(json.dumps({
            "image_path": str(p),
            "emotion": res.get("emotion"),
            "confidence": res.get("confidence"),
            "details": res.get("details")
        }, indent=2))


if __name__ == "__main__":
    main()