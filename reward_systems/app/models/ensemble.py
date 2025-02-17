# model_server/models/ensemble.py
import logging
from typing import List, Dict
from fastapi import HTTPException

from reward.model.inference import RewardModelInference
from sentiment.model.inference import SentimentPredictor
from toxigen.inference import RobertaInference

from config import ModelConfig

logger = logging.getLogger(__name__)

class EnsembleModel:
    def __init__(
        self,
        reward_model_config: ModelConfig,
        sentiment_model_config: ModelConfig,
        roberta_model_config: ModelConfig,
        config_path: str
    ):
        """앙상블 모델 초기화"""
        try:
            # 모델 초기화
            self.reward_model = RewardModelInference(
                reward_model_config.model_path,
                config_path
            )
            self.sentiment_model = SentimentPredictor(
                sentiment_model_config.model_path
            )
            self.roberta_model = RobertaInference(
                roberta_model_config.model_path
            )
            
            # 가중치 정규화
            total_weight = (
                reward_model_config.weight + 
                sentiment_model_config.weight + 
                roberta_model_config.weight
            )
            self.reward_weight = reward_model_config.weight / total_weight
            self.sentiment_weight = sentiment_model_config.weight / total_weight
            self.roberta_weight = roberta_model_config.weight / total_weight
            
            logger.info(
                f"Initialized ensemble with weights: "
                f"Reward={self.reward_weight:.2f}, "
                f"Sentiment={self.sentiment_weight:.2f}, "
                f"RoBERTa={self.roberta_weight:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    async def predict_batch(
        self,
        comments: List[str],
        titles: List[str],
        batch_size: int = 32
    ) -> List[Dict[str, float]]:
        """배치 예측 수행"""
        try:
            # 각 모델 예측
            sentiment_scores = [
                float(score) 
                for score in self.sentiment_model.predict_batch(comments, batch_size)
            ]
            
            roberta_raw_scores = self.roberta_model.predict_batch(
                comments, titles, batch_size
            )
            roberta_scores = [float(scores[1]) for scores in roberta_raw_scores]
            
            # 리워드 모델 예측
            reward_scores = []
            for i in range(0, len(comments), batch_size):
                batch_comments = comments[i:i + batch_size]
                batch_titles = titles[i:i + batch_size]
                batch_scores = []
                
                for comment, title in zip(batch_comments, batch_titles):
                    score = float(self.reward_model.get_reward_score(
                        context=title,
                        comment=comment
                    ))
                    batch_scores.append(score)
                reward_scores.extend(batch_scores)
            
            # 앙상블 스코어 계산
            results = []
            for r_score, s_score, rb_score in zip(
                reward_scores, sentiment_scores, roberta_scores
            ):
                ensemble_score = (
                    self.reward_weight * r_score +
                    self.sentiment_weight * s_score +
                    self.roberta_weight * rb_score
                )
                
                results.append({
                    "reward_score": float(r_score),
                    "sentiment_score": float(s_score),
                    "roberta_score": float(rb_score),
                    "ensemble_score": float(ensemble_score)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def is_healthy(self) -> bool:
        """모델 상태 확인"""
        try:
            # 간단한 추론 테스트
            test_comment = "테스트 문장입니다."
            test_title = "테스트"
            
            self.reward_model.get_reward_score(test_title, test_comment)
            self.sentiment_model.predict_batch([test_comment], 1)
            self.roberta_model.predict_batch([test_comment], [test_title], 1)
            
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False