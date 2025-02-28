import torch
import torch.utils.data
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback
)
import wandb
from datetime import datetime
import yaml
import os
from functools import partial
from torch.utils.data import DataLoader
import logging
from typing import Tuple, Dict, Any

from data_loader import PairwisePreferenceDataset
from utils.trainer import CustomTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def setup_wandb(config):
    run_name = f"reward-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    wandb.init(
        project=config['wandb']['project'],
        name=run_name,
        config={**config['training'], **config['model'], **config['optimization']},
        tags=config['wandb'].get('tags', []),
        notes=config['wandb'].get('notes', '')
    )

def load_model_and_tokenizer(config, pretrained):
    if pretrained:
        logger.info(f"Loading checkpoint from: {pretrained}")
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained,
            local_files_only=True,
            num_labels=config['model']['num_labels']
        )
        tokenizer = AutoTokenizer.from_pretrained(pretrained, local_files_only=True)
    else:
        logger.info(f"Loading base model: {config['model']['name']}")
        model = AutoModelForSequenceClassification.from_pretrained(
            config['model']['name'],
            num_labels=config['model']['num_labels'],
            problem_type=config['model']['problem_type']
        )
        tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

    # GPU 메모리 최적화 설정
    if torch.cuda.is_available():
        model = model.cuda()
        model.gradient_checkpointing_enable()  # 메모리 효율성 개선
        
    return model, tokenizer

def create_dataloaders(dataset, config, tokenizer):
    train_dataset, val_dataset = dataset.get_train_val(
        val_ratio=1 - config['data']['train_val_split']
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['per_device_train_batch_size'],
        shuffle=True,
        num_workers=config['training']['dataloader_num_workers'],
        collate_fn=PairwisePreferenceDataset.collate_fn,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['training']['per_device_eval_batch_size'],
        shuffle=False,
        num_workers=config['training']['dataloader_num_workers'],
        collate_fn=PairwisePreferenceDataset.collate_fn,
        pin_memory=True
    )

    return train_dataloader, val_dataloader

def train_reward_model(data_path, output_dir,
                       config, pretrained=None):
    
    setup_wandb(config)

    model, tokenizer = load_model_and_tokenizer(config, pretrained)
    
    training_args = TrainingArguments(
        output_dir=output_dir, 
        **config['training'],
        do_eval=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # 데이터셋 준비
    dataset = PairwisePreferenceDataset(data_path, tokenizer, config['data'])
    train_dataloader, val_dataloader = create_dataloaders(dataset, config, tokenizer)
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader.dataset,
        eval_dataset=val_dataloader.dataset,
        compute_metrics=None,
        # callbacks=[EarlyStoppingCallback(
        #     early_stopping_patience=config['early_stopping_patience']
        # )]
    )

    logger.info("Starting training...")

    train_result = trainer.train()
    
    # 모델 저장
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    logger.info("Running final evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    return model, tokenizer

def main():
    config_path = "/data/ephemeral/home/reward_systems/reward/config.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 경로 설정
    data_path = config['path']['data']
    output_dir = config['path']['output']
    pretrained_path = config['path'].get('pretrained')

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 학습 실행
    model, tokenizer = train_reward_model(
        data_path,
        output_dir,
        config,
        pretrained=pretrained_path
    )
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
