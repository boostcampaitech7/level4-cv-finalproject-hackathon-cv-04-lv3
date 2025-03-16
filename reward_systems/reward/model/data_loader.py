import torch
from torch.utils.data import Dataset, Subset
import json
import numpy as np
from functools import partial
from typing import Dict, List, Tuple

class PairwisePreferenceDataset(Dataset):
    def __init__(self, json_path, tokenizer, config):
        
        self.tokenizer = tokenizer
        self.max_length = config.get('max_length', 512)
        self.prompt = config['prompt']

        with open(json_path, 'r', encoding='utf-8') as f:
            self.data=json.load(f)['pairs']
        
    
    def _prepare_text(self, context, scripts, response):
        return f"<|prompter|>{self.prompt}\n context: {context}\n scripts: {scripts}<|assistant|>{response}"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        scripts = item['chosen'] + item['rejected']
        chosen_text = self._prepare_text(item['context'], scripts, item['chosen'])
        rejected_text = self._prepare_text(item['context'], scripts, item['rejected'])

        chosen_encodings = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        rejected_encodings = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            "chosen_input_ids": chosen_encodings['input_ids'].squeeze(0),
            "chosen_attention_mask": chosen_encodings['attention_mask'].squeeze(0),
            "rejected_input_ids": rejected_encodings['input_ids'].squeeze(0),
            "rejected_attention_mask": rejected_encodings['attention_mask'].squeeze(0)
        }
    
    @staticmethod
    def collate_fn(batch):
        return {
            "chosen_input_ids": torch.stack([x["chosen_input_ids"] for x in batch]),
            "chosen_attention_mask": torch.stack([x["chosen_attention_mask"] for x in batch]),
            "rejected_input_ids": torch.stack([x["rejected_input_ids"] for x in batch]),
            "rejected_attention_mask": torch.stack([x["rejected_attention_mask"] for x in batch])
        }
    
    def get_train_val(self, val_ratio=0.1, seed=42):
        dataset_size = len(self)
        indices = np.arange(dataset_size)

        np.random.seed(seed)
        np.random.shuffle(indices)

        split_idx = int(dataset_size * (1 - val_ratio))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        return Subset(self, train_indices), Subset(self, val_indices)
