import os
import sys
import json
import wandb
from dataclasses import dataclass
from typing import Dict, List

# 기존 imports와 설정 유지
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from reward.model.inference import RewardModelInference
from sentiment.model.inference import SentimentPredictor
from toxigen.inference import RobertaInference
from weighted_score import EnsemblePredictor, ModelConfig

@dataclass
class ModelConfig:
    model_path: str
    weight: float

def sweep_configuration():
    """wandb sweep 설정 생성"""
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization 사용
        'metric': {
            'name': 'alpha_f1_score',
            'goal': 'maximize'
        },
        'parameters': {
            'economic_alpha': {'min': 0.5, 'max': 2.0},
            'culture_alpha': {'min': 0.5, 'max': 2.0},
            'politics_alpha': {'min': 0.5, 'max': 2.0},
            'regional_alpha': {'min': 0.5, 'max': 2.0},
            'social_alpha': {'min': 0.5, 'max': 2.0},
            'sports_alpha': {'min': 0.5, 'max': 2.0},
            'weather_alpha': {'min': 0.5, 'max': 2.0},
            'world_alpha': {'min': 0.5, 'max': 2.0}
        }
    }
    return sweep_config

def train(config=None):
    """sweep agent가 실행할 학습 함수"""
    with wandb.init(config=config):
        config = wandb.config
        
        # alpha values 구성
        alpha_values = {
            "경제": config.economic_alpha,
            "문화·연예": config.culture_alpha,
            "정치": config.politics_alpha,
            "지역": config.regional_alpha,
            "사회": config.social_alpha,
            "스포츠": config.sports_alpha,
            "날씨": config.weather_alpha,
            "세계": config.world_alpha
        }
        
        # 모델 설정
        BASE_PATH = "/data/ephemeral/home/reward_systems"
        #REWARD_MODEL_PATH = os.path.join(BASE_PATH, "reward/deberta-v2-9_checkpoints")
        SENTIMENT_MODEL_PATH = os.path.join(BASE_PATH, "sentiment/model/create_data_results/checkpoint-2934")
        ROBERTA_MODEL_PATH = os.path.join(BASE_PATH, "ensemble/roberta.tsv")
        ROBERTA_LARGE_MODEL_PATH = os.path.join(BASE_PATH, "ensemble/roberta_large.tsv")
        BERT_MODEL_PATH = os.path.join(BASE_PATH, "ensemble/bert.tsv")
        CONFIG_PATH = os.path.join(BASE_PATH, "reward/config.yaml")
        EVAL_DATA_PATH = os.path.join(BASE_PATH, "ensemble/test_data.json")
        OUTPUT_PATH = os.path.join(BASE_PATH, "ensemble/ensemble22_create_data_results.json")
            
        # 모델 설정
        sentiment_config = ModelConfig(
            model_path=SENTIMENT_MODEL_PATH,
            weight=0.25
        )
        roberta_config = ModelConfig(
            model_path=ROBERTA_MODEL_PATH,
            weight=0.25
        )
        bert_config = ModelConfig(
            model_path=BERT_MODEL_PATH,
            weight=0.25
        )
        roberta_large_config = ModelConfig(
            model_path=ROBERTA_LARGE_MODEL_PATH,
            weight=0.25
        )
        
        # 앙상블 예측기 초기화
        predictor = EnsemblePredictor(
            sentiment_config,
            bert_config,
            roberta_config,
            roberta_large_config,
            CONFIG_PATH,
            alpha_values,
            run_name=f"sweep_trial"
        )
        
        # 데이터 로드
        with open(EVAL_DATA_PATH, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
        
        # 예측 수행
        results = predictor.predict_batch(eval_data)
        
        # wandb에 메트릭 기록은 EnsemblePredictor 내부에서 자동으로 수행됨

def main():
    # sweep 설정 초기화
    sweep_config = sweep_configuration()
    
    # sweep 생성
    sweep_id = wandb.sweep(sweep_config, project="offensive-text-detection")
    
    # agent 실행 (50회 시도)
    wandb.agent(sweep_id, train, count=50)

if __name__ == "__main__":
    main()