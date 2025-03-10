from transformers import (
   AutoModelForSequenceClassification, 
   AutoTokenizer, 
   Trainer, 
   TrainingArguments,
   TrainerCallback, 
   EarlyStoppingCallback
)
from datasets import Dataset
import evaluate
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import wandb
import torch
import json


def load_data(data_path):
    """JSON 데이터 로드 및 train/val 분할"""
    # JSON 파일 읽기
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # JSON 데이터를 DataFrame으로 변환
    df = pd.DataFrame(data)
    
    # train/val 분할
    train_df, val_df = train_test_split(
        df, 
        test_size=0.1, 
        random_state=42, 
        stratify=df['label']
    )
    return train_df, val_df

def preprocess_function(examples, tokenizer):
   """텍스트 전처리 및 토크나이징"""
   return tokenizer(
       examples['text'],
       truncation=True,
       padding=True,
       max_length=512,
       return_tensors=None
   )


def create_datasets(train_df, val_df, tokenizer):
   """train/val 데이터셋 생성"""
   train_dataset = Dataset.from_pandas(train_df)
   val_dataset = Dataset.from_pandas(val_df)

   train_dataset = train_dataset.map(
       lambda x: preprocess_function(x, tokenizer), 
       batched=True
   )
   val_dataset = val_dataset.map(
       lambda x: preprocess_function(x, tokenizer), 
       batched=True
   )
   
   return train_dataset, val_dataset


def compute_metrics(eval_pred):
    """평가 메트릭 계산"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')
    precision = precision_metric.compute(predictions=predictions, references=labels, average='weighted')
    recall = recall_metric.compute(predictions=predictions, references=labels, average='weighted')
    
    return {
        'accuracy': accuracy['accuracy'],
        'f1': f1['f1'],
        'precision': precision['precision'],
        'recall': recall['recall']
    }


class WandbCallback(TrainerCallback):
   """Wandb 로깅을 위한 콜백"""
   def on_log(self, args, state, control, logs=None, **kwargs):
       if state.is_world_process_zero:
           wandb.log(logs)


def prepare_model(model):
   """모델 파라미터 고정 및 학습할 레이어 선택"""
   for name, param in model.named_parameters():
       param.requires_grad = False

   for name, param in model.named_parameters():
       if any(x in name for x in ['layer.4', 'layer.5', 'pre_classifier', 'classifier']):
           param.requires_grad = True
   
   return model


def main():
   # wandb 초기화
   wandb.init(project="reward-model-training", name="sentiment-finetuning")

   # 모델과 토크나이저 로드
   model_name = "tabularisai/multilingual-sentiment-analysis"
   pretrained = "/data/ephemeral/home/reward_systems/sentiment/model/kold_data_results/checkpoint-5690"
   data_path = "/data/ephemeral/home/reward_systems/sentiment/dataset/create_data_sentiment.json"
   output_dir = "./create_data_results"
   
   model = AutoModelForSequenceClassification.from_pretrained(pretrained)
   tokenizer = AutoTokenizer.from_pretrained(pretrained)

   # 데이터 준비
   train_df, val_df = load_data(data_path)
   train_dataset, val_dataset = create_datasets(train_df, val_df, tokenizer)
   
   # 모델 준비
   model = prepare_model(model)

   # 학습 설정
   training_args = TrainingArguments(
       output_dir=output_dir,
       num_train_epochs=3,
       per_device_train_batch_size=32,
       per_device_eval_batch_size=64,
       learning_rate=2e-5,
       weight_decay=0.01,
       save_strategy="steps",         
       save_steps=1000,            
       evaluation_strategy="steps",     
       eval_steps=500,                
       logging_dir="./logs",
       logging_steps=100,               
       gradient_accumulation_steps=2,  
       fp16=True,
       report_to="wandb",
       load_best_model_at_end=True,
       metric_for_best_model="f1",
       warmup_steps=100, 
       lr_scheduler_type="cosine", 
       max_grad_norm=1.0, 
   )

   # 트레이너 초기화
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=train_dataset,
       eval_dataset=val_dataset,
       tokenizer=tokenizer,
       compute_metrics=compute_metrics,
       callbacks=[WandbCallback],
   )

   # 학습 실행
   trainer.train()

   # 모델과 토크나이저 저장
   final_output_dir = f"{output_dir}/final-model"
   trainer.save_model(final_output_dir)
   tokenizer.save_pretrained(final_output_dir)

   # wandb 종료
   wandb.finish()


if __name__ == "__main__":
   main()