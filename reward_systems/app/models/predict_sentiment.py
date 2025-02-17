from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json
from tqdm import tqdm
import pandas as pd

class SentimentPredictor:
   def __init__(self, model_path):
       self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
       self.tokenizer = AutoTokenizer.from_pretrained(model_path)
       self.model.to(self.device)
       self.model.eval()

   def predict_batch(self, texts, batch_size=2):
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
               negative_confidences = predictions[:, 1].cpu().numpy()
               results.extend(negative_confidences)
   
       return results
