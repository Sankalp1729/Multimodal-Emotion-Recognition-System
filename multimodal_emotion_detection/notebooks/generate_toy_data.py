from pathlib import Path
import csv
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

EMOTIONS = ["happy", "sad", "angry", "neutral", "fear", "surprise", "disgust"]

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def ensure_dirs():
    (DATA_DIR / "audio" / "processed" / "clips").mkdir(parents=True, exist_ok=True)
    img_root = DATA_DIR / "image" / "processed"
    for e in EMOTIONS:
        (img_root / e).mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "text").mkdir(parents=True, exist_ok=True)


def generate_audio(duration=1.5, sr=16000, per_class=4):
    labels_path = DATA_DIR / "audio" / "processed" / "labels.csv"
    clips_root = DATA_DIR / "audio" / "processed" / "clips"
    rows = []
    rng = np.random.default_rng(42)

    for e in EMOTIONS:
        for i in range(per_class):
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            # Base tone selection per emotion
            if e == "happy":
                y = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 660 * t)
            elif e == "sad":
                y = 0.4 * np.sin(2 * np.pi * 220 * t)
            elif e == "angry":
                y = 0.3 * np.sin(2 * np.pi * 200 * t) + 0.2 * rng.standard_normal(t.shape)
            elif e == "neutral":
                y = 0.3 * np.sin(2 * np.pi * 330 * t)
            elif e == "fear":
                y = 0.3 * np.sin(2 * np.pi * 330 * t) * (1 + 0.5 * np.sin(2 * np.pi * 5 * t))
            elif e == "surprise":
                y = 0.5 * np.sin(2 * np.pi * 880 * t) * (t < duration * 0.5)
            elif e == "disgust":
                y = 0.3 * np.sign(np.sin(2 * np.pi * 150 * t))
            else:
                y = 0.2 * np.sin(2 * np.pi * 300 * t)
            y = y.astype(np.float32)
            rel = f"clips/{e}_{i}.wav"
            sf.write(str(clips_root / f"{e}_{i}.wav"), y, sr)
            rows.append((rel, e))

    with open(labels_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "label"])
        for r in rows:
            w.writerow(r)


def generate_images(img_size=(224, 224), per_class=6):
    img_root = DATA_DIR / "image" / "processed"
    colors = {
        "happy": (255, 215, 0),      # gold
        "sad": (70, 130, 180),       # steel blue
        "angry": (220, 20, 60),      # crimson
        "neutral": (128, 128, 128),  # gray
        "fear": (138, 43, 226),      # blueviolet
        "surprise": (255, 140, 0),   # dark orange
        "disgust": (34, 139, 34),    # forest green
    }
    for e in EMOTIONS:
        for i in range(per_class):
            img = Image.new("RGB", img_size, colors.get(e, (200, 200, 200)))
            draw = ImageDraw.Draw(img)
            text = f"{e[:3]}-{i}"
            draw.text((10, 10), text, fill=(0, 0, 0))
            img.save(img_root / e / f"{e}_{i}.png")


def generate_text(per_class=8):
    csv_path = DATA_DIR / "text" / "dataset.csv"
    examples = {
        "happy": ["I am thrilled with the results!", "What a wonderful day!", "Feeling great and positive.", "This makes me smile."] ,
        "sad": ["I feel down today.", "This is heartbreaking.", "I miss those times.", "I am not okay."],
        "angry": ["This is infuriating!", "I am so mad right now.", "Stop wasting my time.", "That was unacceptable."],
        "neutral": ["It is what it is.", "I will check the report.", "Please send the document.", "Okay."] ,
        "fear": ["I am worried about this.", "This scares me.", "I feel anxious.", "I am afraid of the outcome."],
        "surprise": ["Wow, I didn't expect that!", "That's shocking!", "Really?", "Unbelievable."],
        "disgust": ["This is gross.", "I can't stand this.", "That's nasty.", "It makes me sick."],
    }
    rows = []
    for e, texts in examples.items():
        for i, t in enumerate(texts):
            rows.append((t, e))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    ensure_dirs()
    generate_audio()
    generate_images()
    generate_text()
    print("Toy datasets generated under data/ for audio, image, and text.")