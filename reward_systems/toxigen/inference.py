import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class RobertaInference:
    def __init__(self, model_path: str):
        """
        BERT 기반 모델을 사용하는 예측기
        Args:
            model_path: 모델 체크포인트 경로
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(
        self,
        texts: List[str],
        contexts: List[str],
        batch_size: int = 32
    ) -> List[float]:
        """
        배치 단위로 예측 수행
        Args:
            texts: 평가할 텍스트 리스트
            contexts: 컨텍스트(제목) 리스트
            batch_size: 배치 크기
        Returns:
            List[float]: 예측 점수 리스트
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_contexts = contexts[i:i + batch_size]
            
            # 각 텍스트와 컨텍스트를 [SEP]으로 결합하여 토큰화
            inputs = self.tokenizer(
                batch_contexts,
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # BERT 계열 모델이면 token_type_ids 추가
            if self.model.config.model_type not in ["roberta"]:
                inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
            
            # 예측 수행
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # sigmoid를 적용하여 0~1 사이의 점수로 변환
            scores = torch.sigmoid(outputs.logits).squeeze(-1).cpu().numpy()
            results.extend(scores.tolist())
        
        return results