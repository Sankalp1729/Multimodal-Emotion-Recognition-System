from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from multimodal_emotion_detection.inference.predict_emotion import predict_emotion

router = APIRouter()

class PredictRequest(BaseModel):
    image_path: Optional[str] = None
    audio_path: Optional[str] = None
    text: Optional[str] = None

@router.post("/predict")
async def predict(req: PredictRequest):
    result = predict_emotion(req.image_path, req.audio_path, req.text)
    return result