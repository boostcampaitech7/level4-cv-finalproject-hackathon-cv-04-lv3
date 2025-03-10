# model_server/schemas/request.py
from pydantic import BaseModel
from typing import List, Dict

class PredictionRequest(BaseModel):
    title_list: List[str]
    text_list: List[str]

class PredictionResponse(BaseModel):
    results: List[float]

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    gpu_available: bool = False  # GPU 사용 가능 여부

