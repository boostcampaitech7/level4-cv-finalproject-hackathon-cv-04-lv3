import os
import sys
import time
import torch
import torchaudio

sys.path.append('submodules/CosyVoice/third_party/Matcha-TTS')
from submodules.CosyVoice.cosyvoice.cli.cosyvoice import CosyVoice2
from submodules.CosyVoice.cosyvoice.utils.file_utils import load_wav

from .processing import *
from api_benchmark import log_time
import asyncio

async def sound_transfer(temp_file_path, scripts):
    total_start_time = time.time()  # 전체 실행 시간 측정 시작
    
    # CosyVoice 모델 로드
    model_load_start = time.time()
    cosyvoice = CosyVoice2('submodules/CosyVoice/pretrained_models/CosyVoice2-0.5B', 
                          load_jit=False, 
                          load_trt=False, 
                          fp16=False)
    model_load_end = time.time()

    results = []

    all_audio_extract_time = 0
    all_video_extract_time = 0
    all_inference_time = 0
    all_save_audio_time = 0

    for script in scripts:
        segment_data = {
            "time_info": {
                "start": script['start'],
                "end": script['end']
            },
            "video_path": None,
            "audio_path": None
        }

        audio_extract_start = time.time()
        audio_segment = await asyncio.to_thread(extract_audio_segment_memory,
            temp_file_path,
            script['start'],
            script['end']
        )
        audio_extract_end = time.time()
        all_audio_extract_time += (audio_extract_end - audio_extract_start)

        video_extract_start = time.time()
        video_segment = await asyncio.to_thread(extract_video_segment,
            temp_file_path,
            script['start'],
            script['end']
        )
        video_extract_end = time.time()
        all_video_extract_time += (video_extract_end - video_extract_start)


        segment_data["video_path"] = video_segment

        inference_start = time.time()
        prompt_speech_16k = load_wav(audio_segment, 16000)

        inference_start = time.time()
        inference_results = await asyncio.to_thread(cosyvoice.inference_zero_shot,
            script['change_text'],
            script['origin_text'],
            prompt_speech_16k,
            stream=False
        )
        inference_end = time.time()
        all_inference_time += (inference_end - inference_start)

        for i, result in enumerate(inference_results):
            output_path = f'/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-04-lv3/cosyvoice_result/audio/output_{script["start"]}.wav'
            
            save_start = time.time()
            await asyncio.to_thread(torchaudio.save,
                output_path,
                result['tts_speech'],
                cosyvoice.sample_rate
            )
            save_end = time.time()
            all_save_audio_time += (save_end - save_start)

            segment_data["audio_path"] = output_path

            results.append(segment_data)
    
    total_end_time = time.time()  # 전체 실행 시간 측정 종료
    log_time('cosyvoice_timelog.txt', f"전체 sound_transfer 실행 시간: {total_end_time - total_start_time:.4f}초")

    log_time('cosyvoice_timelog.txt', f"""
    처리시간 분석
    - Cosyvoice 모델 로딩 시간: {model_load_end - model_load_start:.4f} 초
    - 오디오 추출 시간: {all_audio_extract_time} 초
    - 비디오 추출 시간: {all_video_extract_time} 초
    - CosyVoice 음성 변환 시간: {all_inference_time:.4f} 초
    - 오디오 저장 시간: {all_save_audio_time:.4f} 초""")
    
    return results
