# inference.py

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
import yaml
import json
from typing import List, Dict, Union, Optional
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RewardModelInference:
    def __init__(self, model_path: str, config_path: str):
        """
        리워드 모델 추론을 위한 클래스
        Args:
            model_path: 학습된 모델이 저장된 경로
            config_path: 설정 파일 경로
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 설정 파일 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 모델과 토크나이저 로드
        logger.info(f"Loading model from {model_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()

        # 프롬프트 설정
        self.prompt = self.config['data']['prompt']
        self.max_length = self.config['model']['max_length']

    def _prepare_text(self, context: str, scripts: str, comment: str) -> str:
        """텍스트 준비"""
        return f"<|prompter|>{self.prompt}\n context: {context}\n scripts: {scripts}<|assistant|>{comment}"

    @torch.no_grad()
    def get_reward_score(self, context: str, comment: str, scripts: Optional[str] = None) -> float:
        """
        단일 텍스트에 대한 리워드 점수 계산
        Args:
            context: 컨텍스트 (동영상 제목 등)
            comment: 평가할 텍스트
            scripts: 스크립트 (옵션)
        Returns:
            float: 리워드 점수
        """
        prepared_text = self._prepare_text(context, scripts if scripts else "", comment)
        
        inputs = self.tokenizer(
            prepared_text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze()
        score = 1 - torch.sigmoid(logits).cpu().item()  # sigmoid 적용 후 1에서 빼기
        
        return score

    @torch.no_grad()
    def evaluate_dataset(self, 
                        eval_data_path: str, 
                        output_path: str,
                        batch_size: int = 32) -> Dict:
        """
        평가 데이터셋에 대한 추론 수행
        Args:
            eval_data_path: 평가 데이터셋 경로
            output_path: 결과 저장 경로
            batch_size: 배치 크기
        Returns:
            Dict: 평가 결과 통계
        """
        # 데이터 로드
        with open(eval_data_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)

        results = []
        all_scores = []
        
        # 배치 처리
        for i in tqdm(range(0, len(eval_data), batch_size), desc="Evaluating"):
            batch = eval_data[i:i + batch_size]
            batch_texts = [
                self._prepare_text(item['title'], "", item['comment'])
                for item in batch
            ]
            
            # 토크나이징
            inputs = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            # 추론
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze()
            
            # sigmoid 적용 후 1에서 빼기
            scores = (1 - torch.sigmoid(logits)).cpu()
            
            # 리스트로 변환
            if scores.dim() == 0:  # 단일 스칼라인 경우
                scores = [scores.item()]
            else:
                scores = scores.tolist()
            
            all_scores.extend(scores)
            
            # 결과 저장
            for item, score in zip(batch, scores):
                true_value = 1.0 if item['OFF'] else 0.0
                results.append({
                    **item,
                    "reward_score": score,
                    "true_value": true_value
                })

        metrics = self._calculate_metrics(scores, true_value)
        
        # 결과 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "results": results,
                "metrics": metrics
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"Evaluation completed. Results saved to {output_path}")
        logger.info(f"Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

        return metrics
    
    def _calculate_metrics(scores, true_values, threshold: float = 0.5):
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
                f"accuracy": accuracy,
                f"precision": precision,
                f"recall": recall,
                f"f1_score": f1,
                f"true_positives": int(tp),
                f"false_positives": int(fp),
                f"true_negatives": int(tn),
                f"false_negatives": int(fn)
            }

def main():
    # 설정
    MODEL_PATH = "/data/ephemeral/home/reward_systems/reward/result-epoch20/checkpoint-3450"
    CONFIG_PATH = "/data/ephemeral/home/reward_systems/reward/config.yaml"
    EVAL_DATA_PATH = "/data/ephemeral/home/reward_systems/ensemble/eval_data.json"
    OUTPUT_PATH = "/data/ephemeral/home/reward_systems/reward/evaluation_results_epoch5.json"
    
    # 리워드 모델 초기화
    reward_model = RewardModelInference(MODEL_PATH, CONFIG_PATH)
    
    # 데이터셋 평가
    metrics = reward_model.evaluate_dataset(
        EVAL_DATA_PATH,
        OUTPUT_PATH,
        batch_size=32
    )

if __name__ == "__main__":
    main()