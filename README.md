# Multimodal Emotion Detection

A production-shaped FastAPI service and CLI for multimodal emotion recognition (image, audio, text) with optional learned fusion. Includes a containerized deployment and configurable model artifact locations.

## Quickstart (Local)

1) Create a virtual environment and install dependencies
- PowerShell (Windows):
  - python -m venv .venv
  - .\.venv\Scripts\Activate.ps1
  - pip install -r multimodal_emotion_detection/requirements.txt

2) Run the API locally
- uvicorn multimodal_emotion_detection.api.app:app --host 127.0.0.1 --port 8000 --workers 1

3) Verify endpoints
- Health: curl http://127.0.0.1:8000/health
- Readiness: curl http://127.0.0.1:8000/ready

4) Make predictions
- JSON (paths on local disk):
  - curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"audio_path\": \"c:/path/to/audio.wav\"}"
- Multipart (file upload):
  - curl -X POST http://127.0.0.1:8000/predict -F "audio_file=@c:/path/to/audio.wav;type=audio/wav"
- Multipart (public URLs):
  - curl -X POST http://127.0.0.1:8000/predict -F "image_url=https://example.com/face.jpg" -F "text=Hello there!"

Optional: CLI inference (no web server):
- python -m multimodal_emotion_detection.run --audio "c:/path/to/audio.wav"

## Quickstart (Docker)

Prerequisite: Docker Desktop installed and running.

1) Build the image
- docker build -t mme-api:latest .

2) Run the container (PowerShell)
- .\run_container.ps1 -ImageName "mme-api:latest" -Port 8000 -ModelsHostDir "$PWD\multimodal_emotion_detection\models" -TextAllowDownload false -Workers 1
  - Mounts host models at /models inside the container
  - Sets PORT and UVICORN_WORKERS for the API process

3) Verify and predict
- Health: curl http://127.0.0.1:8000/health
- Readiness: curl http://127.0.0.1:8000/ready
- Predict (JSON): curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\": \"hello world\"}"

## Environment Variables

Configuration is centralized in multimodal_emotion_detection/utils/config.py. You can override defaults via environment variables:

- MME_BASE_DIR: Base directory for the package (default: project package root)
- MME_DATA_DIR: Dataset root (default: <BASE_DIR>/data)
- MME_MODELS_DIR: Models root (default: <BASE_DIR>/models, or /models inside the container)
- MME_AUDIO_MODEL_DIR: Audio model directory (default: <MME_MODELS_DIR>/audio_model)
- MME_IMAGE_MODEL_DIR: Image model directory (default: <MME_MODELS_DIR>/image_model)
- MME_FUSION_MODEL_DIR: Fusion model directory (default: <MME_MODELS_DIR>/fusion_model)
- MME_TEXT_MODEL_ID: HuggingFace model ID for text (default: cardiffnlp/twitter-roberta-base-sentiment)
- MME_TEXT_MODEL_DIR: Local directory to cache or pre-bake the text model (default: <MME_MODELS_DIR>/text_hf)
- MME_TEXT_ALLOW_DOWNLOAD: Enable internet downloads for text model at runtime ("1"/"true" to allow; default disabled)

Container runtime knobs:
- PORT: API port (default: 8000)
- UVICORN_WORKERS: Number of worker processes (default: 1)

## Model Artifact Layout

Place model artifacts under MME_MODELS_DIR (on host by default at ./multimodal_emotion_detection/models; inside the container at /models). Each modality expects specific files:

- audio_model/
  - model.pt: Torch state_dict for the AudioMLP
  - scaler.pkl: StandardScaler fitted on training features
  - label_map.json: mapping of emotion label -> class index used in training
  - meta.json: metadata including input_dim, temperature, num_params, val_acc, feature_columns

- image_model/
  - resnet18.pt: Torch state_dict for a ResNet-18 classifier head
  - label_map.json: mapping of class_name -> index
  - meta.json: metadata such as architecture and val_acc

- fusion_model/
  - model.pkl: sklearn classifier (e.g., logistic regression) saved via pickle
  - scaler.pkl: optional sklearn scaler saved via pickle
  - meta.json: metadata including emotions and feature_names

- text_hf/
  - Pre-baked HuggingFace model cache for TEXT_MODEL_ID (optional if MME_TEXT_ALLOW_DOWNLOAD is enabled; otherwise required)

If an expected artifact is missing, the system uses safe fallbacks:
- Audio: heuristic features with a lightweight MLP (predicts with reduced confidence)
- Image: falls back to basic heuristics or returns low-confidence distribution
- Fusion: uses weighted-average fusion with entropy-aware confidence

## Project Files of Interest

- API: multimodal_emotion_detection/api/app.py (endpoints: /health, /ready, /predict)
- Predictors: multimodal_emotion_detection/inference/predict_audio.py, predict_image.py, predict_text.py, fusion.py
- Config: multimodal_emotion_detection/utils/config.py
- Container helper: run_container.ps1
- Dependencies: multimodal_emotion_detection/requirements.txt

## Notes

- For Windows PowerShell, paths use backslashes; ensure you escape or quote paths when using curl.
- Large datasets and models are excluded from version control by .gitignore. Mount them for Docker via -v and place them locally under multimodal_emotion_detection/models when running outside Docker.