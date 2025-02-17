# utils/trainer.py
from transformers import Trainer
import torch
import wandb
from typing import Dict, Tuple, Any, Optional
import numpy as np
import logging

from .optimization import LayerwiseLearningRateDecay, GradualUnfreezing

logger = logging.getLogger(__name__)

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._signature_columns = [
            "chosen_input_ids",
            "chosen_attention_mask",
            "rejected_input_ids",
            "rejected_attention_mask"
        ]

        self.gradual_unfreezing = GradualUnfreezing(
            self.model, 
            total_steps = len(self.train_dataset) * self.args.num_train_epochs,
            warmup_steps=1000,
            min_steps_per_layer=500
        )
        self.current_step = 0
        self.margin_warmup_steps = 1000
        self.initial_margin = 1.0
        self.final_margin = 0.7

        self._init_metrics_tracking()
    
    def _init_metrics_tracking(self):
        self.training_metrics = {
            'grad_norms': [],
            'layer_metrics': {},
            'optimization_metrics': {}
        }

    def create_optimizer(self):
        decay = LayerwiseLearningRateDecay(
            self.model, self.args.learning_rate, 
            min_rate=0.5, max_rate=0.95
        )
        self.optimizer = torch.optim.AdamW(
            decay.parameter_groups,
            weight_decay=self.args.weight_decay
        )
        return self.optimizer
    
    def get_dynamic_margin(self):
        if self.current_step < self.margin_warmup_steps:
            progress = self.current_step / self.margin_warmup_steps
            margin = self.initial_margin - (self.initial_margin - self.final_margin) * progress
        else:
            margin = self.final_margin
        return margin
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self.current_step += 1
        batch_size = inputs["chosen_input_ids"].size(0)
        
        combined_input_ids = torch.cat([
            inputs["chosen_input_ids"],
            inputs["rejected_input_ids"]
        ], dim=0)
        combined_attention_mask = torch.cat([
            inputs["chosen_attention_mask"],
            inputs["rejected_attention_mask"]
        ], dim=0)
        
        outputs = model(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask,
        )

        logits = outputs.logits.squeeze(-1)
        chosen_reward = logits[:batch_size]
        rejected_reward = logits[batch_size:]
        
        margin = self.get_dynamic_margin()
        reward_diff = chosen_reward - rejected_reward
        loss_scale = torch.exp(torch.abs(reward_diff).detach())
        
        loss = torch.max(
            torch.zeros_like(chosen_reward),
            (-reward_diff + margin) * loss_scale
        ).mean()

        if return_outputs:
            return loss, (outputs, reward_diff)
        return loss
    
    def training_step(self, *args, **kwargs):
        self.gradual_unfreezing.update(self.state.global_step)

        with torch.cuda.amp.autocast(enabled=self.args.fp16):
            loss = super().training_step(*args, **kwargs)

        if self.state.global_step % self.args.logging_steps == 0:
            self._log_training_metrics()

        return loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys = None):
        # 배치 처리 최적화
        batch_size = inputs["chosen_input_ids"].size(0)
        
        # 하나의 배치로 합치기
        combined_input_ids = torch.cat([
            inputs["chosen_input_ids"],
            inputs["rejected_input_ids"]
        ], dim=0)
        
        combined_attention_mask = torch.cat([
            inputs["chosen_attention_mask"],
            inputs["rejected_attention_mask"]
        ], dim=0)

        with torch.no_grad():
            outputs = model(
                input_ids=combined_input_ids,
                attention_mask=combined_attention_mask,
            )
            all_logits = outputs.logits.squeeze(-1)
            
            ## 결과 분리
            chosen_logits = all_logits[:batch_size]
            rejected_logits = all_logits[batch_size:]

            if prediction_loss_only:
                # Compute loss
                margin = self.get_dynamic_margin()
                loss = torch.max(
                    torch.zeros_like(chosen_logits),
                    -chosen_logits + rejected_logits + margin
                ).mean()
                return (loss, None, None)
            
            return (None, (chosen_logits, rejected_logits), None)

    def _log_training_metrics(self):
        metrics = {
            'train/current_margin': self.get_dynamic_margin(),
            'train/unfrozen_layers': len([p for p in self.model.parameters() if p.requires_grad])
        }

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                metrics[f"grad_norm/{name}"] = param.grad.norm().item()

        for idx, group in enumerate(self.optimizer.param_groups):
            metrics[f"train/lr_group_{idx}"] = group["lr"]

        wandb.log(metrics, step=self.state.global_step)
    
    def compute_metrics(self, eval_pred)-> Dict[str, float]:
        chosen_logits = eval_pred.predictions[0]
        rejected_logits = eval_pred.predictions[1]
        
        rewards_chosen = chosen_logits.squeeze(-1)
        rewards_rejected = rejected_logits.squeeze(-1)

        rewards_chosen = np.array(rewards_chosen)
        rewards_rejected = np.array(rewards_rejected)

        margin = self.get_dynamic_margin()
        loss = np.maximum(
            np.zeros_like(rewards_chosen),
            -rewards_chosen + rewards_rejected + margin
        ).mean()

        metrics = {
            'eval_loss': loss,
            'eval_accuracy': (rewards_chosen > rewards_rejected).mean(),
            'eval_margin_violations': (rewards_chosen - rewards_rejected < margin).mean(),
            'eval_avg_reward_diff': (rewards_chosen - rewards_rejected).mean(),

            'eval_chosen_mean': rewards_chosen.mean(),
            'eval_chosen_std': rewards_chosen.std(),
            'eval_rejected_mean': rewards_rejected.mean(),
            'eval_rejected_std': rewards_rejected.std(),
            
            'current_margin': self.get_dynamic_margin(),
            'unfrozen_layers': len([p for p in self.model.parameters() if p.requires_grad])
        }
        
        wandb.log({
            "eval/loss": metrics['eval_loss'],
            "eval/accuracy": metrics['eval_accuracy'],
            "eval/margin_violations": metrics['eval_margin_violations'],
            "eval/avg_reward_diff": metrics['eval_avg_reward_diff'],
            "eval/chosen_rewards_hist": wandb.Histogram(rewards_chosen),
            "eval/rejected_rewards_hist": wandb.Histogram(rewards_rejected),
            # 리워드 차이 분포
            "eval/reward_diff_hist": wandb.Histogram((rewards_chosen - rewards_rejected)),
        }, step=self.state.global_step)

        logger.info(f"Evaluation metrics at step {self.state.global_step}:")
        logger.info(f"  Loss: {metrics['eval_loss']:.4f}")
        logger.info(f"  Accuracy: {metrics['eval_accuracy']:.4f}")
        logger.info(f"  Margin Violations: {metrics['eval_margin_violations']:.4f}")
        logger.info(f"  Average Reward Difference: {metrics['eval_avg_reward_diff']:.4f}")

        return metrics
    
