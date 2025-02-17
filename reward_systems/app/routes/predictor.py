# model_server/routes/predictor.py
import torch
from fastapi import APIRouter, HTTPException
from typing import List

from schemas.request import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse
)
from models.ensemble import EnsembleModel
from models.predict_sentiment import SentimentPredictor
from config import settings

router = APIRouter()

# 모델 인스턴스 초기화
# model = EnsembleModel(
#     reward_model_config=settings.REWARD_MODEL,
#     sentiment_model_config=settings.SENTIMENT_MODEL,
#     roberta_model_config=settings.ROBERTA_MODEL,
#     config_path=settings.CONFIG_PATH
# )

model = SentimentPredictor(settings.SENTIMENT_MODEL.model_path)

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """예측 엔드포인트"""
    if len(request.text_list) != len(request.title_list):
        raise HTTPException(
            status_code=400,
            detail="Number of comments and titles must match"
        )
    
    results = model.predict_batch(
        texts=request.text_list,
        batch_size=settings.BATCH_SIZE
    )
    
    return PredictionResponse(results=results)

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스체크 엔드포인트"""
    models_loaded = model.is_healthy()
    gpu_available = torch.cuda.is_available()
    
    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        models_loaded=models_loaded,
        gpu_available=gpu_available
    )