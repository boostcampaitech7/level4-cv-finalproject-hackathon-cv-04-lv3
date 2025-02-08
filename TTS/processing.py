import ffmpeg
import numpy as np

def extract_audio_segment(input_file, start_time, end_time):
    """
    비디오에서 특정 구간의 오디오를 추출
    """
    start_seconds = start_time / 1000
    duration = (end_time - start_time) / 1000
    
    output_file = f'temp_{start_time}.wav'
    process = (
        ffmpeg
        .input(input_file)
        .output(output_file,
                format='wav',
                acodec='pcm_s16le',
                ar=16000,
                ac=1,
                ss=start_seconds,
                t=duration)
        .run()
    )
    
    return output_file

def extract_video_segment(input_file, start_time, end_time):
    """
    비디오에서 특정 구간을 추출
    """
    start_seconds = start_time / 1000
    duration = (end_time - start_time) / 1000
    
    output_file = f'temp_video_{start_time}.mp4'
    process = (
        ffmpeg
        .input(input_file, ss=start_seconds, t=duration)
        .output(output_file,
                codec='copy')  # 코덱 복사로 빠른 처리
        .run()
    )
    
    return output_file