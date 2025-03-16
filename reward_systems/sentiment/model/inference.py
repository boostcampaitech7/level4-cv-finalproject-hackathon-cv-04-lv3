from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

class SentimentPredictor:
   def __init__(self, model_path):
       self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
       self.tokenizer = AutoTokenizer.from_pretrained(model_path)
       self.model.to(self.device)
       self.model.eval()

   def predict_batch(self, texts, batch_size=32):
       results = []
       
       for i in tqdm(range(0, len(texts), batch_size)):
           batch_texts = texts[i:i + batch_size]
           inputs = self.tokenizer(
               batch_texts,
               truncation=True,
               padding=True,
               max_length=512,
               return_tensors="pt"
           )
           inputs = {k: v.to(self.device) for k, v in inputs.items()}
           
           with torch.no_grad():
               outputs = self.model(**inputs)
               predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
               
               # Negative(1) 클래스에 대한 confidence만 추출
               very_negative_confidences = predictions[:, 0].cpu().numpy()
               negative_confidences = predictions[:, 1].cpu().numpy()
               results.extend(negative_confidences+very_negative_confidences)
   
       return results


def main():
   # 경로 설정
   model_path = "/data/ephemeral/home/reward_systems/sentiment/model/kold_data_results/checkpoint-5690"
   test_path = "/data/ephemeral/home/reward_systems/ensemble/test_data.json"
   output_path = "inference_results.csv"
   
   # JSON 데이터 로드
   with open(test_path, 'r', encoding='utf-8') as f:
       test_data = json.load(f)
   
   # 텍스트와 라벨 추출
   texts = [item['comment'] for item in test_data]
   #true_labels = [item['OFF'] for item in test_data]
   true_labels = [1 if item['OFF'] else 0 for item in test_data]

   # 예측기 초기화
   predictor = SentimentPredictor(model_path)
   
   # 예측 수행
   negative_confidences = predictor.predict_batch(texts)
   
   # 예측값 변환 (threshold = 0.5)
   predicted_labels = [1 if conf > 0.5 else 0 for conf in negative_confidences]

   # F1 Score 및 Accuracy 계산
   accuracy = accuracy_score(true_labels, predicted_labels)
   f1 = f1_score(true_labels, predicted_labels)
   
   # 결과 데이터프레임 생성
   results_df = pd.DataFrame({
       'text': texts,
       'true_label': true_labels,
       'predicted_label': predicted_labels,
       'negative_confidence': negative_confidences
   })
   
   # 결과 저장
#    results_df.to_csv(output_path, index=False)
   
   # 통계 출력
   print("\nInference Results Summary:")
   print(f"Total samples: {len(results_df)}")
   print(f"Accuracy: {accuracy:.4f}")
   print(f"F1 Score: {f1:.4f}")
   print(f"Average Negative confidence: {results_df['negative_confidence'].mean():.4f}")
   print(f"High confidence predictions (>0.8): {(results_df['negative_confidence'] > 0.8).sum()}")
   
   # True Negative 샘플에 대한 성능
   true_negatives = results_df[results_df['true_label'] == 1]
   print(f"\nTrue Negative samples: {len(true_negatives)}")
   print(f"Average confidence for True Negatives: {true_negatives['negative_confidence'].mean():.4f}")
   
   # 몇 가지 예시 출력
   print("\nSample Predictions:")
   for _, row in results_df.head(5).iterrows():
       print(f"\nText: {row['text']}")
       print(f"True Label: {row['true_label']}")
       print(f"Predicted Label: {row['predicted_label']}")
       print(f"Negative Confidence: {row['negative_confidence']:.4f}")


if __name__ == "__main__":
   main()