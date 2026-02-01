import json
from pathlib import Path
import sys

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
from multimodal_emotion_detection.utils.config import DATA_DIR


def main():
    url = "http://127.0.0.1:8000/predict"
    clips_dir = DATA_DIR / "audio" / "processed" / "clips"
    paths = sorted(list(clips_dir.glob("*.wav")))[:3]
    print("Posting audio files:", [str(p) for p in paths])
    with httpx.Client(timeout=30.0) as client:
        for p in paths:
            payload = {"audio_path": str(p)}
            resp = client.post(url, json=payload)
            try:
                data = resp.json()
            except Exception:
                data = {"status_code": resp.status_code, "text": resp.text}
            print(json.dumps({
                "request": payload,
                "response": data
            }, indent=2))


if __name__ == "__main__":
    main()