import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from pydub import AudioSegment

class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x
    
class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits

device = 'cpu'
load_path = "/data/ephemeral/home/hackathon/level4-cv-finalproject-hackathon-cv-04-lv3/Emotion/model"
processor = Wav2Vec2Processor.from_pretrained(load_path)
model = EmotionModel.from_pretrained(load_path).to(device)
    

def process_func_batch(
        audio_inputs: list,
        sampling_rate: int = 16000,
        embeddings: bool = False,
) -> list:
    """
    여러 개의 오디오 데이터를 배치 처리하여 감정 분석 수행
    """
    # 🔹 Feature Extraction
    inputs = processor(audio_inputs, sampling_rate=sampling_rate, padding=True, return_tensors="pt")

    # 🔹 모델 추론 (Batch 처리)
    with torch.no_grad():
        outputs = model(inputs.input_values.float().to(device))
        batch_results = outputs[0] if embeddings else outputs[1]

    batch_results = batch_results.detach().cpu().numpy().tolist()

    # 🔹 결과 저장
    # results = [{"Arousal": result[0], "Dominance": result[1], "Valence": result[2]} for result in batch_results]
    results = [1 if result[0] > 0.7 else 0 for result in batch_results]


    return results