import re
import numpy as np
from langchain.schema import Document
import ffmpeg

def extract_curse_words(text):
    pattern = r'\[(\d+),\s*(\d+),\s*\'([^\']+)\',\s*\'([^\']+)\'\]'
    matches = re.findall(pattern, text)
    
    curse_words = []
    for match in matches:
        start, end, word, category = match
        curse_words.append([int(start), int(end), word, category])
    
    return curse_words

# Clova Speech api에서 받은 결과를 원하는 입력 형태로 전처리 & 500 토큰씩 나눠서 반환
def preprocess_speech_data(speech_data):
    segments = speech_data["segments"]
    input_text = ""

    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        input_text += f"[{start}, {end}, '{text}']\n"

    input_text = input_text[:-2] # 마지막 줄바꿈 제거

    documents = [Document(page_content=input_text)]
    return documents

def preprocess_STT_data(speech_data):
    segments = speech_data["segments"]

    formatted_segments = [
        {"start": segment["start"], "end": segment["end"], "text": segment["text"]}
        for segment in segments
    ]

    return formatted_segments

def merge_segments(segments):
    if not segments:
        return []
        
    merged = []
    current = list(segments[0])
    
    for i in range(1, len(segments)):
        start, end, word, filtered_word = segments[i]
        
        if current[1] == start:
            current[1] = end
            current[2] = current[2] + ' ' + word
            current[3] = current[3] + ' ' + filtered_word
        else:
            merged.append(current)
            current = list(segments[i])
    
    merged.append(current)
    return merged

def fade_in_out(fade_duration, main_sr, first_part, last_part, insert_array):
    # Fade In/Out 적용
    # 페이드 인/아웃의 샘플 수 계산
    fade_in_samples = int(fade_duration * main_sr)   # 페이드 인 샘플 수
    fade_out_samples = int(fade_duration * main_sr)  # 페이드 아웃 샘플 수
    
    # 페이드 인/아웃 곡선 생성
    fade_in = np.linspace(0, 1, fade_in_samples)     # 0에서 1로 증가
    fade_out = np.linspace(1, 0, fade_out_samples)   # 1에서 0으로 감소

    # 각 부분에 페이드 적용
    # 첫 번째 파트의 끝부분에 페이드 아웃 적용
    first_part[-fade_out_samples:] *= fade_out

    # 마지막 파트의 시작 부분에 페이드 인 적용
    last_part[:fade_in_samples] *= fade_in

    # 삽입할 데이터의 앞부분에 페이드 인, 뒷부분에 페이드 아웃 적용
    insert_array[:fade_in_samples] *= fade_in
    insert_array[-fade_out_samples:] *= fade_out

def convert_with_ffmpeg_python(input_file):
    """
    MP4 → WAV 변환 후 파일 저장 없이 NumPy 배열로 반환
    """
    process = (
        ffmpeg
        .input(input_file)
        .output("pipe:", format="wav", acodec="pcm_s16le", ar=16000, ac=1)  # WAV 포맷 설정
        .run(capture_stdout=True, capture_stderr=True)  # 메모리에서 직접 처리
    )

    audio_data = np.frombuffer(process[0][44:], dtype=np.int16)  # 바이트 데이터를 NumPy 배열로 변환
    return audio_data

def slice_audio_numpy(audio_data, time_list, sample_rate=16):
    """
    NumPy 배열에서 직접 오디오를 슬라이싱하여 반환
    """
    results = []
    for segment in time_list:
        start_sample = int(segment["start"] * sample_rate)  # 초 → 샘플 인덱스 변환
        end_sample = int(segment["end"] * sample_rate)
        sliced_audio = audio_data[start_sample:end_sample]  # NumPy 슬라이싱
        results.append(sliced_audio)
    return results
