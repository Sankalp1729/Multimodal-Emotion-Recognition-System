import json
from pathlib import Path
import sys

# Ensure project root is on sys.path when running as a script inside the package
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multimodal_emotion_detection.inference.predict_emotion import predict_emotion
from multimodal_emotion_detection.utils.config import DATA_DIR


def main():
    clips_dir = DATA_DIR / "audio" / "processed" / "clips"
    paths = sorted(list(clips_dir.glob("*.wav")))[:3]
    print("Using audio files:", [str(p) for p in paths])
    for p in paths:
        res = predict_emotion(None, str(p), None)
        print(json.dumps({
            "audio_path": str(p),
            "emotion": res.get("emotion"),
            "confidence": res.get("confidence"),
            "details": res.get("details")
        }, indent=2))


if __name__ == "__main__":
    main()