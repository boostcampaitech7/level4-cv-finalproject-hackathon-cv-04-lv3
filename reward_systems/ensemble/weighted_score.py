import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import sys
import torch
import json
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import logging
from dataclasses import dataclass
import wandb
import pandas as pd
import ast


# 프로젝트 루트 경로 설정 및 모델 디렉토리 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# 각 모델의 inference 모듈 import
from reward.model.inference import RewardModelInference
from sentiment.model.inference import SentimentPredictor
from toxigen.inference import RobertaInference

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    model_path: str
    weight: float

class EnsemblePredictor:
    def __init__(
        self,
        #reward_model_config: ModelConfig,
        sentiment_model_config: ModelConfig,
        bert_model_config: ModelConfig,
        roberta_model_config: ModelConfig,
        roberta_large_model_config: ModelConfig,
        config_path: str,
        alpha: Dict,
        run_name: str = "ensemble_evaluation"
    ):
        """
        앙상블 예측을 위한 클래스
        Args:
            reward_model_config: 리워드 모델 설정
            sentiment_model_config: 감성 분석 모델 설정
            roberta_model_config: RoBERTa 모델 설정
            config_path: 리워드 모델용 설정 파일 경로
            run_name: wandb run 이름
        """
        self.alpha = alpha
        # wandb 초기화
        wandb.init(
            project="offensive-text-detection",
            name=run_name,
            config={
                "sentiment_model_path": sentiment_model_config.model_path,
                "roberta_model_path": roberta_model_config.model_path,
                "bert_model_path": bert_model_config.model_path,
                "roberta_large_model_path": roberta_large_model_config.model_path,
                #"reward_weight": reward_model_config.weight,
                "sentiment_weight": sentiment_model_config.weight,
                "roberta_weight": roberta_model_config.weight,
                "bert_weight": bert_model_config.weight,
                "roberta_large_weight": roberta_large_model_config.weight,
                "alpha": alpha
            }
        )
        
        try:
            # self.reward_model = RewardModelInference(
            #     reward_model_config.model_path,
            #     config_path
            # )
            # self.sentiment_model = SentimentPredictor(
            #     sentiment_model_config.model_path
            # )
            # self.roberta_model = RobertaInference(
            #     roberta_model_config.model_path
            # )
            self.sentiment_model = SentimentPredictor(sentiment_model_config.model_path)
            self.roberta_model = roberta_model_config.model_path
            self.bert_model = bert_model_config.model_path
            self.roberta_large_model = roberta_large_model_config.model_path
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
        
        # # 가중치 정규화
        # total_weight = (
        #     reward_model_config.weight + 
        #     sentiment_model_config.weight + 
        #     roberta_model_config.weight
        # )
        # self.reward_weight = reward_model_config.weight / total_weight
        # self.sentiment_weight = sentiment_model_config.weight / total_weight
        # self.roberta_weight = roberta_model_config.weight / total_weight
        # 가중치 정규화
        total_weight = (
            #reward_model_config.weight + 
            sentiment_model_config.weight + 
            roberta_model_config.weight +
            bert_model_config.weight +
            roberta_large_model_config.weight
        )
        #self.reward_weight = reward_model_config.weight / total_weight
        self.sentiment_weight = sentiment_model_config.weight / total_weight
        self.bert_weight = bert_model_config.weight / total_weight
        self.roberta_weight = roberta_model_config.weight / total_weight
        self.roberta_large_weight = roberta_large_model_config.weight / total_weight
        
        logger.info(
            f"Initialized ensemble with weights: "
            #f"Reward={self.reward_weight:.2f}, "
            f"Sentiment={self.sentiment_weight:.2f}, "
            f"RoBERTa={self.roberta_weight:.2f}, "
            f"BERT={self.bert_weight:.2f}, "
            f"RoBERTa_large={self.roberta_large_weight:.2f}, "
        )

    def softmax(self, x):
        """
        소프트맥스 함수를 계산합니다.
        입력된 값들을 0-1 사이의 확률 분포로 변환합니다.
        """
        exp_x = np.exp(x - np.max(x))  # 수치적 안정성을 위해 최대값을 빼줍니다
        return exp_x / exp_x.sum()

    def process_tsv(self, input_file):
        df = pd.read_csv(input_file, sep='\t')
        df['logit'] = df['logit'].apply(ast.literal_eval)
        df['probabilities'] = df['logit'].apply(self.softmax)
        offensive_probs = df['probabilities'].apply(lambda x: x[1]).tolist()
        return offensive_probs

    def predict_batch(
        self,
        eval_data: List[Dict],
        batch_size: int = 32
    ) -> Dict:
        """
        배치 단위로 앙상블 예측 수행
        Args:
            eval_data: 평가할 데이터 리스트
            batch_size: 배치 크기
        Returns:
            Dict: 예측 결과
        """
        texts = [item['comment'] for item in eval_data]
        contexts = [item['title'] for item in eval_data]
        
        # 각 모델 예측
        sentiment_scores = [float(score) for score in self.sentiment_model.predict_batch(texts, batch_size)]
        #roberta_raw_scores = self.roberta_model.predict_batch(texts, contexts, batch_size)
        roberta_scores = self.process_tsv(self.roberta_model)
        bert_scores = self.process_tsv(self.bert_model)
        roberta_large_scores = self.process_tsv(self.roberta_large_model)

        # 리워드 모델 예측
        # reward_scores = []
        # for i in tqdm(range(0, len(eval_data), batch_size), desc="Reward Model Prediction"):
        #     batch = eval_data[i:i + batch_size]
        #     batch_scores = []
        #     for item in batch:
        #         score = float(self.reward_model.get_reward_score(
        #             context=item['title'],
        #             comment=item['comment']
        #         ))
        #         batch_scores.append(score)
        #     reward_scores.extend(batch_scores)
        
        # 앙상블 스코어 계산
        # ensemble_scores = [
        #     self.reward_weight * r + self.sentiment_weight * s + self.roberta_weight * rb
        #     for r, s, rb in zip(reward_scores, sentiment_scores, roberta_scores)
        # ]
        ensemble_scores = [
            self.bert_weight * b + self.sentiment_weight * s + self.roberta_weight * r + self.roberta_large_weight + rl
            for b, s, r, rl in zip(bert_scores, sentiment_scores, roberta_scores, roberta_large_scores)
        ]
        
        # 결과 저장
        # results = []
        # for item, r_score, s_score, rb_score, e_score in zip(
        #     eval_data, reward_scores, sentiment_scores, roberta_scores, ensemble_scores
        # ):
        #     alpha_score = self._calculate_alpha(item['category'], e_score)
        #     results.append({
        #         **item,
        #         "reward_score": float(r_score),
        #         "sentiment_score": float(s_score),
        #         "roberta_score": float(rb_score),
        #         "ensemble_score": float(e_score),
        #         "alpha_score": float(alpha_score),
        #         "true_value": float(1.0 if item['OFF'] else 0.0)
        #     })

        results = []
        for item, b_score, s_score, r_score, rl_score, e_score in zip(
            eval_data, bert_scores, sentiment_scores, roberta_scores, roberta_large_scores, ensemble_scores
        ):
            alpha_score = self._calculate_alpha(item['category'], e_score)
            results.append({
                **item,
                "bert_score": float(b_score),
                "sentiment_score": float(s_score),
                "roberta_score": float(r_score),
                "roberta_large_score": float(rl_score),
                "ensemble_score": float(e_score),
                "alpha_score": float(alpha_score),
                "true_value": float(1.0 if item['OFF'] else 0.0)
            })
        
        # 성능 지표 계산 및 wandb에 기록
        metrics = self._calculate_metrics(results)
        wandb.log(metrics)
        
        return {"results": results}
    
    def _calculate_alpha(self, category, e_score):
        alpha_value = self.alpha[category]
        return alpha_value * e_score


    def _calculate_metrics(
        self,
        results: List[Dict],
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        각 모델의 성능 지표 계산
        Args:
            results: 예측 결과 리스트
            threshold: 분류를 위한 임계값
        Returns:
            Dict[str, float]: 계산된 성능 지표
        """
        def calculate_model_metrics(scores, true_values, prefix=""):
            y_pred = [1 if score >= threshold else 0 for score in scores]
            
            tp = sum(1 for t, p in zip(true_values, y_pred) if t == 1 and p == 1)
            fp = sum(1 for t, p in zip(true_values, y_pred) if t == 0 and p == 1)
            fn = sum(1 for t, p in zip(true_values, y_pred) if t == 1 and p == 0)
            tn = sum(1 for t, p in zip(true_values, y_pred) if t == 0 and p == 0)
            
            accuracy = float((tp + tn) / len(true_values))
            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
            
            return {
                f"{prefix}accuracy": accuracy,
                f"{prefix}precision": precision,
                f"{prefix}recall": recall,
                f"{prefix}f1_score": f1,
                f"{prefix}true_positives": int(tp),
                f"{prefix}false_positives": int(fp),
                f"{prefix}true_negatives": int(tn),
                f"{prefix}false_negatives": int(fn)
            }
        
        true_values = [r["true_value"] for r in results]
        metrics = {}
        
        # 각 모델별 메트릭 계산
        model_configs = {
            "bert": [r["bert_score"] for r in results],
            "sentiment": [r["sentiment_score"] for r in results],
            "roberta": [r["roberta_score"] for r in results],
            "roberta_large": [r["roberta_large_score"] for r in results],
            "ensemble": [r["ensemble_score"] for r in results],
            "alpha": [r["alpha_score"] for r in results]
        }
        
        for model_name, scores in model_configs.items():
            model_metrics = calculate_model_metrics(scores, true_values, f"{model_name}_")
            metrics.update(model_metrics)
            
            # 콘솔에 출력
            logger.info(f"\n{model_name.capitalize()} Model Metrics:")
            for metric_name, value in model_metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")
        
        return metrics

def main():
    # 설정
    BASE_PATH = "/data/ephemeral/home/reward_systems"
    
    # 모델 경로
    SENTIMENT_MODEL_PATH = os.path.join(BASE_PATH, "sentiment/model/create_data_results/checkpoint-2934")
    ROBERTA_MODEL_PATH = os.path.join(BASE_PATH, "ensemble/roberta.tsv")
    ROBERTA_LARGE_MODEL_PATH = os.path.join(BASE_PATH, "ensemble/roberta_large.tsv")
    BERT_MODEL_PATH = os.path.join(BASE_PATH, "ensemble/bert.tsv")
    CONFIG_PATH = os.path.join(BASE_PATH, "reward/config.yaml")
    EVAL_DATA_PATH = os.path.join(BASE_PATH, "ensemble/test_data.json")
    OUTPUT_PATH = os.path.join(BASE_PATH, "ensemble/ensemble22_create_data_results.json")
    
    # 경로 로깅
    logger.info(f"Loading from paths:")
#    logger.info(f"Reward model: {REWARD_MODEL_PATH}")
    logger.info(f"Sentiment model: {SENTIMENT_MODEL_PATH}")
    logger.info(f"RoBERTa model: {ROBERTA_MODEL_PATH}")
    logger.info(f"BERT model: {BERT_MODEL_PATH}")
    logger.info(f"RoBERTa Large model: {ROBERTA_LARGE_MODEL_PATH}")
    logger.info(f"Config: {CONFIG_PATH}")
    
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

    
    # roberta-data2 기준
    alpha_values = [
        {
            "경제": 1.5, 
            "문화·연예": 1.25, 
            "정치": 0.95,
            "지역": 1.8,
            "사회": 1.2,
            "스포츠": 1.85,
            "날씨": 1.2,
            "세계": 1.05,
        }, 
        {
            "경제": 1.0, 
            "문화·연예": 1.3, 
            "정치": 1.0,
            "지역": 0.95,
            "사회": 1.0,
            "스포츠": 1.65,
            "날씨": 1.15,
            "세계": 0.8,
        }, 
        {
            "경제": 1.95, 
            "문화·연예": 1.0, 
            "정치": 1.95,
            "지역": 1.95,
            "사회": 1.95,
            "스포츠": 1.95,
            "날씨": 1.95,
            "세계": 0.5,
        }, 
        {
            "경제": 1.95, 
            "문화·연예": 1.95, 
            "정치": 1.95,
            "지역": 1.95,
            "사회": 1.95,
            "스포츠": 1.95,
            "날씨": 1.95,
            "세계": 1.95,
        }, 
    ]
    
    # 앙상블 예측기 초기화
    predictor = EnsemblePredictor(
        sentiment_config,
        bert_config,
        roberta_config,
        roberta_large_config,
        CONFIG_PATH,
        alpha_values[1],
        #run_name=f"ensemble_w{reward_config.weight:.1f}_{sentiment_config.weight:.1f}"
        run_name=f"ensemble_w{bert_config.weight:.1f}_{sentiment_config.weight:.1f}_{roberta_config.weight:.1f}_{roberta_large_config.weight:.1f}"
    )
    
    # 데이터 로드
    with open(EVAL_DATA_PATH, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    # 예측 수행
    results = predictor.predict_batch(eval_data)
    
    # 결과 저장
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # wandb 종료
    wandb.finish()

if __name__ == "__main__":
    main()