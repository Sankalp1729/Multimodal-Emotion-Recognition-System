# Minimal production-ready Dockerfile for the Multimodal Emotion Detection API
# Uses Python 3.11 slim base and installs system libs needed for audio/image processing

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Default to pre-baked models only; override if you want to allow downloads
    MME_TEXT_ALLOW_DOWNLOAD=false \
    # Default models directory inside the container; mount host models here
    MME_MODELS_DIR=/models \
    # Uvicorn runtime defaults
    PORT=8000 \
    UVICORN_WORKERS=1

# System dependencies for audio (libsndfile via soundfile) and general media
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (use the full dependency set inside the project)
COPY multimodal_emotion_detection/requirements.txt ./multimodal_emotion_detection/requirements.txt
RUN pip install --no-cache-dir -r multimodal_emotion_detection/requirements.txt

# Copy the application source
COPY . .

EXPOSE 8000

# Start the API with sensible defaults
CMD ["sh", "-c", "uvicorn multimodal_emotion_detection.api.app:app --host 0.0.0.0 --port ${PORT} --workers ${UVICORN_WORKERS}"]