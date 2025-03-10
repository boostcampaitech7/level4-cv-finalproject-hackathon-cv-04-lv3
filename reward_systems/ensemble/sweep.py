# sweep.py
import wandb
import yaml
import os
import json
from weighted_ensemble import EnsemblePredictor, ModelConfig

# Sweep 설정
sweep_config = {
    'method': 'bayes',  # Bayesian optimization 사용
    'metric': {
        'name': 'ensemble_f1_score',
        'goal': 'maximize'
    },
    'parameters': {
        'bert_weight': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 1.0
        },
        'sentiment_weight': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 1.0
        },
        'roberta_weight': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 1.0
        },
        'roberta_large_weight': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 1.0
        },

    }
}

def train():
    # wandb 실행 초기화
    wandb.init()
    
    # 설정값 가져오기
    BASE_PATH = "/data/ephemeral/home/reward_systems"
    # REWARD_MODEL_PATH = os.path.join(BASE_PATH, "reward/deberta-v2-9_checkpoints")
    # SENTIMENT_MODEL_PATH = os.path.join(BASE_PATH, "sentiment/model/create_data_results/checkpoint-2934")
    # ROBERTA_MODEL_PATH = os.path.join(BASE_PATH, "toxigen/create_data/checkpoint-3000")
    # CONFIG_PATH = os.path.join(BASE_PATH, "reward/config.yaml")
    # EVAL_DATA_PATH = os.path.join(BASE_PATH, "ensemble/test_data.json")
    SENTIMENT_MODEL_PATH = os.path.join(BASE_PATH, "sentiment/model/create_data_results/checkpoint-2934")
    ROBERTA_MODEL_PATH = os.path.join(BASE_PATH, "ensemble/roberta.tsv")
    ROBERTA_LARGE_MODEL_PATH = os.path.join(BASE_PATH, "ensemble/roberta_large.tsv")
    BERT_MODEL_PATH = os.path.join(BASE_PATH, "ensemble/bert.tsv")
    CONFIG_PATH = os.path.join(BASE_PATH, "reward/config.yaml")
    EVAL_DATA_PATH = os.path.join(BASE_PATH, "ensemble/test_data.json")
    OUTPUT_PATH = os.path.join(BASE_PATH, "ensemble/ensemble22_create_data_results.json")
    
    # wandb 설정에서 가중치 가져오기
    config = wandb.config
    
    # 가중치 정규화
    total_weight = config.bert_weight + config.sentiment_weight + config.roberta_weight + config.roberta_large_weight
    normalized_weights = {
        #'reward': config.reward_weight / total_weight,
        'bert': config.bert_weight / total_weight,
        'sentiment': config.sentiment_weight / total_weight,
        'roberta': config.roberta_weight / total_weight,
        'roberta_large': config.roberta_large_weight / total_weight,
    }
    
    # 모델 설정
    # reward_config = ModelConfig(
    #     model_path=REWARD_MODEL_PATH,
    #     weight=normalized_weights['reward']
    # )
    sentiment_config = ModelConfig(
        model_path=SENTIMENT_MODEL_PATH,
        weight=normalized_weights['sentiment']
    )
    roberta_config = ModelConfig(
        model_path=ROBERTA_MODEL_PATH,
        weight=normalized_weights['roberta']
    )
    bert_config = ModelConfig(
        model_path=BERT_MODEL_PATH,
        weight=normalized_weights['bert']
    )
    roberta_large_config = ModelConfig(
        model_path=ROBERTA_LARGE_MODEL_PATH,
        weight=normalized_weights['roberta_large']
    )
    
    # 앙상블 예측기 초기화
    predictor = EnsemblePredictor(
        sentiment_config,
        bert_config,
        roberta_config,
        roberta_large_config,
        CONFIG_PATH,
        #run_name=f"ensemble_w{reward_config.weight:.1f}_{sentiment_config.weight:.1f}"
        run_name=f"ensemble_w{bert_config.weight:.1f}_{sentiment_config.weight:.1f}_{roberta_config.weight:.1f}_{roberta_large_config.weight:.1f}"
    )
    
    # 앙상블 예측기 초기화 (wandb init은 EnsemblePredictor 내부에서 처리됨)
    # predictor = EnsemblePredictor(
    #     reward_config,
    #     sentiment_config,
    #     roberta_config,
    #     CONFIG_PATH,
    #     run_name=f"sweep_w{normalized_weights['reward']:.2f}_{normalized_weights['sentiment']:.2f}_{normalized_weights['roberta']:.2f}"
    # )
    
    # 평가 데이터 로드
    with open(EVAL_DATA_PATH, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    # 예측 수행
    results = predictor.predict_batch(eval_data)
    
    # wandb 종료
    wandb.finish()

def main():
    # sweep 설정 저장
    with open('sweep_config.yaml', 'w') as f:
        yaml.dump(sweep_config, f)
    
    # sweep 초기화 및 실행
    sweep_id = wandb.sweep(sweep_config, project="offensive-text-detection")
    wandb.agent(sweep_id, train, count=50)  # 50회 시도

if __name__ == "__main__":
    main()