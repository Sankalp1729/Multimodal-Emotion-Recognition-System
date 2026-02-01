import cv2
import numpy as np
import soundfile as sf
import librosa
from typing import Optional

from mod_image import image_emotion_probs
from mod_audio import audio_emotion_probs
from mod_text import text_emotion_probs
from fusion import fuse_probs
from core import EmotionResult

# Demo CLI: provide paths to image and audio, and optional text
# python main.py --image path/to.jpg --audio path/to.wav --text "I am so excited!"

import argparse


def load_image(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    return img


def load_audio(path: str):
    y, sr = sf.read(path)
    # convert to mono float32
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = y.astype(np.float32)
    return y, sr


def run(image_path: Optional[str], audio_path: Optional[str], text_input: Optional[str]) -> EmotionResult:
    modality_probs = {}
    if image_path:
        img = load_image(image_path)
        modality_probs["image"] = image_emotion_probs(img)
    if audio_path:
        y, sr = load_audio(audio_path)
        modality_probs["audio"] = audio_emotion_probs(y, sr)
    if text_input:
        modality_probs["text"] = text_emotion_probs(text_input)

    result = fuse_probs(modality_probs)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--audio", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    args = parser.parse_args()

    res = run(args.image, args.audio, args.text)
    print(res.to_dict())


if __name__ == "__main__":
    main()