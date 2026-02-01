import argparse
from inference.predict_emotion import predict_emotion

# CLI wrapper to run unified inference from the structured project


def main():
    parser = argparse.ArgumentParser(description="Multimodal Emotion Detection CLI")
    parser.add_argument("--image", type=str, default=None, help="Path to face image (optional)")
    parser.add_argument("--audio", type=str, default=None, help="Path to speech audio (optional)")
    parser.add_argument("--text", type=str, default=None, help="Text input (optional)")
    args = parser.parse_args()
    res = predict_emotion(args.image, args.audio, args.text)
    print(res)


if __name__ == "__main__":
    main()