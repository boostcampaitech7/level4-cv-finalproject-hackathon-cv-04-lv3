from typing import List, Dict
import torch
from weakref import proxy
import math

class LayerwiseLearningRateDecay:
    """
    계층적 학습률 적용
    """
    def __init__(self, model, lr, min_rate=0.5, max_rate=0.95):
        self.lr = lr
        self.min_rate = min_rate
        self.max_rate = max_rate
        # 레이어별 파라미터 그룹 생성
        self.parameter_groups = self._create_parameter_groups(model)
    
    # def _create_parameter_groups(self, model):

    #     groups = []
    #     if self.base_model == "deberta":
    #         base_model = model.deberta
    #     elif self.base_model == "electra":
    #         base_model = model.electra
    #     num_layers = base_model.config.num_hidden_layers

    #     def get_decay_rate(layer_idx):
    #         progress = layer_idx / num_layers
    #         return self.min_rate + (self.max_rate - self.min_rate) * (1 - progress)
        
    #     # Embedding layer
    #     groups.append({
    #         "params": list(base_model.embeddings.parameters()),
    #         "lr": self.lr * get_decay_rate(num_layers)
    #     })
        
    #     # Transformer layers
    #     for layer_idx in range(num_layers):
    #         layer = base_model.encoder.layer[layer_idx]
    #         groups.append({
    #             "params": list(layer.parameters()),
    #             "lr": self.lr * get_decay_rate(num_layers - layer_idx - 1)
    #         })

    #     groups.extend([
    #         {
    #         "params": list(base_model.rel_embeddings.parameters()),
    #         "lr": self.lr * get_decay_rate(num_layers // 2)
    #         },
    #         {
    #         "params": list(model.pooler.parameters()),
    #         "lr": self.lr * get_decay_rate(num_layers // 4)
    #         },
    #         {
    #         "params": list(model.classifier.parameters()),
    #         "lr": self.lr
    #         }
    #     ])

    #     return groups
    
    def _create_parameter_groups(self, model):
        groups = []
        
        # 기본 모델 구조 파악
        if hasattr(model, 'deberta'):
            base_model = model.deberta
            model_type = 'deberta'
        elif hasattr(model, 'electra'):
            base_model = model.electra
            model_type = 'electra'
        else:
            raise ValueError("Unsupported model architecture")

        num_layers = base_model.config.num_hidden_layers

        def get_decay_rate(layer_idx):
            progress = layer_idx / num_layers
            return self.min_rate + (self.max_rate - self.min_rate) * (1 - progress)

        # Embeddings layer
        if hasattr(base_model, 'embeddings'):
            groups.append({
                "params": list(base_model.embeddings.parameters()),
                "lr": self.lr * get_decay_rate(num_layers)
            })

        # Transformer layers
        if hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'layer'):
            for layer_idx in range(num_layers):
                layer = base_model.encoder.layer[layer_idx]
                groups.append({
                    "params": list(layer.parameters()),
                    "lr": self.lr * get_decay_rate(num_layers - layer_idx - 1)
                })

        # Model specific components
        if model_type == 'deberta' and hasattr(base_model.encoder, 'rel_embeddings'):
            groups.append({
                "params": list(base_model.encoder.rel_embeddings.parameters()),
                "lr": self.lr * get_decay_rate(num_layers // 2)
            })

        # Pooler (if exists)
        if hasattr(model, 'pooler') and model.pooler is not None:
            groups.append({
                "params": list(model.pooler.parameters()),
                "lr": self.lr * get_decay_rate(num_layers // 4)
            })

        # Classifier layer
        if hasattr(model, 'classifier'):
            groups.append({
                "params": list(model.classifier.parameters()),
                "lr": self.lr
            })
        elif hasattr(model, 'score'):  # Some models use 'score' instead of 'classifier'
            groups.append({
                "params": list(model.score.parameters()),
                "lr": self.lr
            })

        # Verify that all parameters are included
        all_params = set()
        for group in groups:
            all_params.update(id(p) for p in group["params"])
        
        # Add any remaining parameters with base learning rate
        remaining_params = [
            p for p in model.parameters() 
            if id(p) not in all_params
        ]
        if remaining_params:
            groups.append({
                "params": remaining_params,
                "lr": self.lr
            })

        return groups

class GradualUnfreezing:
    """
    점진적으로 unfreezing
    """
    def __init__(self, model, total_steps, warmup_steps=1000, 
                 min_steps_per_layer=500):
        self.model = model
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_steps_per_layer = min_steps_per_layer

        if hasattr(model, 'deberta'):
            self.base_model = model.deberta
            self.model_type = 'deberta'
        elif hasattr(model, 'electra'):
            self.base_model = model.electra
            self.model_type = 'electra'
        else:
            raise ValueError("Unsupported model architecture")

        self.layers = list(self.base_model.encoder.layer)
        self.num_layers = len(self.layers)

        self._initialize_frozen_state()
        self.unfreeze_schedule = self._create_schedule()
    
    def _initialize_frozen_state(self):
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Classifier 레이어 unfrozen
        if hasattr(self.model, 'classifier'):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'score'):  # Some models use 'score' instead
            for param in self.model.score.parameters():
                param.requires_grad = True

        # Pooler 레이어 unfrozen (있는 경우)
        if hasattr(self.model, 'pooler') and self.model.pooler is not None:
            for param in self.model.pooler.parameters():
                param.requires_grad = True

    def _create_schedule(self):
        total_trainable_steps = self.total_steps - self.warmup_steps
        steps_per_layer = max(
            self.min_steps_per_layer,
            total_trainable_steps // self.num_layers
        )

        return {
            self.warmup_steps + (i * steps_per_layer): self.num_layers - i - 1
            for i in range(self.num_layers)
        }

    def update(self, step):
        if step < self.warmup_steps:
            return
        
        for schedule_step, layer_idx in sorted(self.unfreeze_schedule.items()):
            if step >= schedule_step:
                self._unfreeze_up_to_layer(layer_idx)

    def _unfreeze_up_to_layer(self, target_layer_idx):
        # Transformer 레이어 unfreezing
        for idx in range(self.num_layers-1, target_layer_idx-1, -1):
            for param in self.layers[idx].parameters():
                param.requires_grad=True
        # 모델의 절반 이상이 unfrozen되면 임베딩 레이어들도 unfreeze
        if target_layer_idx >= self.num_layers // 2:
            if hasattr(self.base_model, 'embeddings'):
                for param in self.base_model.embeddings.parameters():
                    param.requires_grad = True

            # 모델별 특수 임베딩 처리
            if self.model_type == 'deberta':
                if hasattr(self.base_model.encoder, 'rel_embeddings'):
                    for param in self.base_model.encoder.rel_embeddings.parameters():
                        param.requires_grad = True
            elif self.model_type == 'electra':
                if hasattr(self.base_model, 'embeddings'):
                    for param in self.base_model.embeddings.parameters():
                        param.requires_grad = True
