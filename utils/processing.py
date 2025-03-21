import re
import numpy as np
from langchain_core.documents import Document
import ffmpeg
import feedparser

from .llm import get_solar_pro
from prompts import load_template

# Vector DB 저장 형식에 맞게 뉴스 전처리
def extract_rss_content(link):

   parse_rss = feedparser.parse(link)
   documents = []
   llm = get_solar_pro(max_token=5000, temperature=0.0)

   for entry in parse_rss.entries:
        content = entry.get('content', [{}])[0].get('value', '')
        messages = [{"role": "user", "content": load_template("extract_rss_content_prompt", content)}]
        response = llm.invoke(messages)
        summary = response.content

        document = Document(
            page_content=summary,
            metadata={
                'published': entry.get('published', ''),
                'category': entry.get('tags', [{}])[0].get('term', ''),
                'title': entry.get('title', ''),
                'link': entry.get('link', '')
            }
        )
        documents.append(document)
   return documents

# Clova Speech api에서 받은 결과를 [{"start"=int, "end"=int, "text"=str] 형태로 전처리
def preprocess_STT_data(speech_data, separators=[".", "?"]):
    formatted_segments = []
    
    current_start = None
    current_sentence = []
    for segment in speech_data['segments']:
        for word in segment['words']:
            if current_start is None:
                current_start = word[0]
            current_sentence.append(word[2])
            
            if any(separator in word[2] for separator in separators):
                formatted_segments.append({"start": current_start, "end": word[1], "text": ' '.join(current_sentence)})
                current_start = None
                current_sentence = []
                
    return formatted_segments

# Document 요소를 가지는 list로 변환
def preprocess_script_items(script_items):
    result = []
    
    for item in script_items:
        result.append(f"[{item.start}, {item.end}, '{item.text}']")
        
    return [Document(page_content='\n'.join(result))]

def parse_response(response):
    try:
        response_str = response['result']
        # 문자열에서 리스트 형태의 부분들을 추출
        pattern = r'\[([^]]+)\]'
        matches = re.findall(pattern, response_str)
        
        results = []
        for match in matches:
            in_pattern = r'<([^>]+)>'
            
            elements = re.findall(in_pattern, match)
            if len(elements) >= 5:
                # 숫자 문자열을 정수로 변환
                start_time = int(elements[0].strip())
                end_time = int(elements[1].strip())
                # 문자열에서 따옴표 제거
                original_text = elements[2].strip()
                explanation = elements[3].strip()
                suggested_text = elements[4].strip()
                
                results.append({
                    'start': start_time,
                    'end': end_time,
                    'origin_text': original_text,
                    'new_text': suggested_text,
                    'reason' : explanation,
                    'title': response['source_documents'][0].metadata['title']
                })
        
        return results
    except Exception as e:
        print(f"파싱 오류: {e}")
        return []

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