import os
import sys
sys.path.append('submodules/CosyVoice/third_party/Matcha-TTS')
from submodules.CosyVoice.cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from submodules.CosyVoice.cosyvoice.utils.file_utils import load_wav
import torch
import torchaudio

from .processing import extract_audio_segment

def sound_transfer(temp_file_path, scripts):
    # CosyVoice 모델 로드
    cosyvoice = CosyVoice2('submodules/CosyVoice/pretrained_models/CosyVoice2-0.5B', 
                          load_jit=False, 
                          load_trt=False, 
                          fp16=False)
    
    results = []
    for script in scripts:
        # 구간 오디오 추출
        audio_segment = extract_audio_segment(
            temp_file_path,
            script['start'],
            script['end']
        )
        
        prompt_speech_16k = load_wav(audio_segment, 16000)
        
        # CosyVoice 처리
        inference_results = cosyvoice.inference_zero_shot(
            script['change_text'],
            script['origin_text'],
            prompt_speech_16k,
            stream=False
        )
        
        # 결과 저장
        for i, result in enumerate(inference_results):
            output_path = f'output_{script["start"]}_{i}.wav'
            torchaudio.save(
                output_path,
                result['tts_speech'],
                cosyvoice.sample_rate
            )
            results.append(output_path)
    return results