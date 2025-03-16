import io
import ffmpeg
import numpy as np
import cv2    
import tempfile
import os

def extract_audio_segment(input_file, start_time, end_time):
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

def extract_audio_segment_memory(input_file, start_time, end_time):
    """
    비디오 특정 구간의 오디오를 메모리에서 직접 처리하여 반환 
    """
    
    start_seconds = start_time / 1000
    duration = (end_time - start_time) / 1000
    
    # BytesIO 객체를 임시 파일로 저장 (ffmpeg-python에서 BytesIO 객체를 받지 못하는 듯)
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        temp_path = temp_file.name
        input_file.seek(0)
        temp_file.write(input_file.read())
    
    try:
        # ffmpeg 명령어 실행
        process = (
            ffmpeg
            .input(temp_path, ss=start_seconds, t=duration)
            .output('pipe:', format='wav', acodec='pcm_s16le', ar=16000, ac=1)
            .run(capture_stdout=True, capture_stderr=True)
        )
    finally:
        # 임시 파일 삭제
        os.unlink(temp_path)
    
    audio_data = io.BytesIO(process[0])
    return audio_data

def extract_video_segment(input_file, start_time, end_time):
    """
    비디오에서 특정 구간을 추출
    """
    start_seconds = start_time / 1000
    duration = (end_time - start_time) / 1000
    
    output_file = f'/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-04-lv3/cosyvoice_result/face/temp_video_{start_time}.mp4'
    process = (
        ffmpeg
        .input(input_file, ss=start_seconds, t=duration)
        .output(output_file,
                codec='copy'
                )  # 코덱 복사로 빠른 처리
        .overwrite_output() 
        .run()
    )
    return output_file


def extract_video_segment_memory(input_file, start_time, end_time):
    """
    비디오에서 특정 구간을 메모리에 추출
    """
    
    start_seconds = start_time / 1000
    duration = (end_time - start_time) / 1000
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        temp_path = temp_file.name
        input_file.seek(0)
        temp_file.write(input_file.read())
    
    try:
        # ffmpeg 명령어 실행
        process = (
            ffmpeg
            .input(temp_path, ss=start_seconds, t=duration)
            .output('pipe:', format='mp4', movflags='frag_keyframe+empty_moov')
            .run(capture_stdout=True, capture_stderr=True)
        )
    finally:
        # 임시 파일 삭제
        os.unlink(temp_path)
    
    # BytesIO 객체로 변환
    video_data = io.BytesIO(process[0])
    return video_data

def extract_video_segment_opencv(input_file, start_time, end_time):
    """
    OpenCV를 사용하여 특정 구간의 비디오를 메모리 처리
    """
    cap = cv2.VideoCapture(input_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time / 1000 * fps)
    end_frame = int(end_time / 1000 * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # OpenCV는 BGR이므로 RGB 변환

    cap.release()
    return np.array(frames)

def save_video_opencv(video_frames, output_path, fps=30):
    """
    numpy 배열 형태의 비디오 데이터를 mp4 파일로 저장
    """
    if len(video_frames) == 0:
        print("비디오 프레임이 없습니다.")
        return

    height, width, _ = video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 코덱 설정
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in video_frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # OpenCV는 BGR 포맷이므로 변환 필요

    out.release()
    print(f"비디오 저장 완료: {output_path}")