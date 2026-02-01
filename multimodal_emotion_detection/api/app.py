from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from typing import Optional
import tempfile
import os
import requests
import numpy as np

app = FastAPI(title="Multimodal Emotion Detection API")

# Allowed MIME types and size limits (bytes)
IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
AUDIO_MIME_TYPES = {"audio/wav", "audio/x-wav", "audio/wave"}
TEXT_MIME_TYPES = {"text/plain"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_AUDIO_BYTES = 20 * 1024 * 1024  # 20 MB
MAX_TEXT_BYTES = 32 * 1024  # 32 KB

_SUFFIX_MAP = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
}

async def _save_upload_to_temp(upload: UploadFile, allowed_types: set[str], max_bytes: int) -> str:
    if not upload:
        raise HTTPException(status_code=400, detail="No file provided.")
    content_type = (upload.content_type or "").split(";")[0].strip().lower()
    if content_type not in allowed_types:
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {content_type}. Allowed: {sorted(allowed_types)}")
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Limit: {max_bytes} bytes")
    suffix = _SUFFIX_MAP.get(content_type, "")
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(tmp_path, "wb") as f:
        f.write(data)
    return tmp_path

async def _read_text_upload(upload: UploadFile, allowed_types: set[str], max_bytes: int) -> str:
    if not upload:
        raise HTTPException(status_code=400, detail="No file provided.")
    content_type = (upload.content_type or "").split(";")[0].strip().lower()
    if content_type not in allowed_types:
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {content_type}. Allowed: {sorted(allowed_types)}")
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded text file is empty.")
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Text file too large. Limit: {max_bytes} bytes")
    try:
        text_value = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise HTTPException(status_code=415, detail="Text file must be UTF-8 encoded.")
    return text_value

def _download_url_to_temp(url: str, allowed_types: set[str], max_bytes: int) -> str:
    if not url or not url.strip():
        raise HTTPException(status_code=400, detail="URL must be a non-empty string.")
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")
    try:
        resp = requests.get(url, stream=True, timeout=(3, 10))
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: HTTP {resp.status_code}")
    content_type = (resp.headers.get("content-type", "").split(";")[0].strip().lower())
    if content_type not in allowed_types:
        raise HTTPException(status_code=415, detail=f"Unsupported media type from URL: {content_type}. Allowed: {sorted(allowed_types)}")
    content_length = resp.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > max_bytes:
                raise HTTPException(status_code=413, detail=f"Remote file too large. Limit: {max_bytes} bytes")
        except ValueError:
            # Ignore invalid content-length
            pass
    suffix = _SUFFIX_MAP.get(content_type, "")
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    total = 0
    try:
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                    raise HTTPException(status_code=413, detail=f"Remote file too large during download. Limit: {max_bytes} bytes")
                f.write(chunk)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise
    return tmp_path

@app.post("/predict")
async def predict(
    request: Request,
    image_file: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    text_file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    audio_url: Optional[str] = Form(None),
):
    # Lazy import to avoid heavy initialization during app startup
    from multimodal_emotion_detection.inference.predict_emotion import predict_emotion

    # Backward compatibility: accept JSON {image_path, audio_path, text}
    content_type_header = (request.headers.get("content-type", "").lower())
    if content_type_header.startswith("application/json"):
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body.")
        image_path = payload.get("image_path")
        audio_path = payload.get("audio_path")
        text_value = payload.get("text")
        if not any([image_path, audio_path, text_value]):
            raise HTTPException(status_code=422, detail="Provide at least one modality: image_path, audio_path, or text")
        result = predict_emotion(image_path, audio_path, text_value)
        return result

    # New: multipart/form-data with file uploads and/or public URLs
    image_tmp = None
    audio_tmp = None
    text_value = None
    try:
        # Prefer uploaded files over URLs if both provided
        if image_file is not None:
            image_tmp = await _save_upload_to_temp(image_file, IMAGE_MIME_TYPES, MAX_IMAGE_BYTES)
        elif image_url:
            image_tmp = _download_url_to_temp(image_url, IMAGE_MIME_TYPES, MAX_IMAGE_BYTES)

        if audio_file is not None:
            audio_tmp = await _save_upload_to_temp(audio_file, AUDIO_MIME_TYPES, MAX_AUDIO_BYTES)
        elif audio_url:
            audio_tmp = _download_url_to_temp(audio_url, AUDIO_MIME_TYPES, MAX_AUDIO_BYTES)

        if text_file is not None:
            text_value = await _read_text_upload(text_file, TEXT_MIME_TYPES, MAX_TEXT_BYTES)
        elif text is not None:
            # Enforce size limit on form text
            encoded = text.encode("utf-8", errors="ignore")
            if len(encoded) > MAX_TEXT_BYTES:
                raise HTTPException(status_code=413, detail=f"Text too long. Limit: {MAX_TEXT_BYTES} bytes")
            text_value = text

        if not any([image_tmp, audio_tmp, text_value]):
            raise HTTPException(status_code=422, detail="Provide at least one modality: image_file/image_url, audio_file/audio_url, or text/text_file")

        result = predict_emotion(image_tmp, audio_tmp, text_value)
        return result
    finally:
        # Clean up any temporary files we created
        for path in (image_tmp, audio_tmp):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/ready")
async def ready():
    checks = {
        "text_tokenizer": False,
        "audio_model_loaded": False,
        "image_model_loaded": False,
        "fusion_model_loaded": False,
        "dry_run_ok": False,
    }
    errors = {}

    # Check text tokenizer/model availability via a small inference
    try:
        from multimodal_emotion_detection.inference.predict_text import predict_text_probs
        probs_text = predict_text_probs("ok")
        checks["text_tokenizer"] = isinstance(probs_text, dict) and len(probs_text) > 0
    except Exception as e:
        errors["text"] = str(e)

    # Check audio model loaded (trained artifacts) if available
    try:
        from multimodal_emotion_detection.inference.predict_audio import _ensure_audio_model_loaded, predict_audio_probs
        checks["audio_model_loaded"] = bool(_ensure_audio_model_loaded())
        # Also verify audio inference path works (trained or heuristic)
        y = np.zeros(16000, dtype=np.float32)
        probs_audio = predict_audio_probs(y, 16000)
        checks["dry_run_ok"] = checks["dry_run_ok"] or (isinstance(probs_audio, dict) and len(probs_audio) > 0)
    except Exception as e:
        errors["audio"] = str(e)

    # Check image model loaded (trained artifacts) if available
    try:
        from multimodal_emotion_detection.inference.predict_image import _ensure_image_model_loaded
        checks["image_model_loaded"] = bool(_ensure_image_model_loaded())
    except Exception as e:
        errors["image"] = str(e)

    # Check fusion model loaded (learned fusion), may be optional
    try:
        from multimodal_emotion_detection.inference.fusion import _ensure_fusion_model_loaded
        checks["fusion_model_loaded"] = bool(_ensure_fusion_model_loaded())
    except Exception as e:
        errors["fusion"] = str(e)

    # Dry-run end-to-end inference (text-only) using public API
    try:
        from multimodal_emotion_detection.inference.predict_emotion import predict_emotion
        res = predict_emotion(None, None, "hello world")
        checks["dry_run_ok"] = checks["dry_run_ok"] or (isinstance(res, dict) and "emotion" in res and "confidence" in res)
    except Exception as e:
        errors["dry_run"] = str(e)

    # Define readiness policy: tokenizer available AND dry-run succeeded; models loaded when present
    ready_ok = checks["text_tokenizer"] and checks["dry_run_ok"]

    if not ready_ok:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail={"status": "not_ready", "checks": checks, "errors": errors})

    return {"status": "ready", "checks": checks}