# model_server/config.py
import os
from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_path: str
    weight: float

class Settings:
    # 기본 경로 설정
    BASE_PATH = "/data/ephemeral/home/reward_systems"
    
    # 모델 설정
    REWARD_MODEL = ModelConfig(
        model_path=os.path.join(BASE_PATH, "reward/deberta-v2-9_checkpoints"),
        weight=0.2
    )
    SENTIMENT_MODEL = ModelConfig(
        model_path=os.path.join(BASE_PATH, "sentiment/model/create_data_results/checkpoint-2934"),
        weight=0.4
    )
    ROBERTA_MODEL = ModelConfig(
        model_path=os.path.join(BASE_PATH, "toxigen/checkpoint-2000"),
        weight=0.4
    )
    
    # 설정 파일 경로
    CONFIG_PATH = os.path.join(BASE_PATH, "reward/config.yaml")
    
    # 서버 설정
    HOST = "0.0.0.0"
    PORT = 32157
    
    # 기타 설정
    BATCH_SIZE = 2

settings = Settings()