from fastapi import FastAPI, Request
import uvicorn
import subprocess
import os
import io
import base64
import tempfile
from moviepy.editor import VideoFileClip, concatenate_videoclips
from fastapi.responses import Response, StreamingResponse
import time

app = FastAPI()

def insert_processed_video_memory(whole_video_data, processed_video_data, start_time, end_time):
    """
    메모리 내에서 영상 처리: 원본 영상에서 start_time 이전 부분과 end_time 이후 부분을 추출한 뒤,
    그 사이에 processed_video를 삽입하여 메모리에 저장합니다.
    """
    
    # 밀리초를 초로 변환
    start_sec = start_time / 1000
    end_sec = end_time / 1000

    # 임시 파일 생성
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as whole_temp:
        whole_temp_path = whole_temp.name
        if isinstance(whole_video_data, io.BytesIO):
            whole_video_data.seek(0)
            whole_temp.write(whole_video_data.read())
        else:
            with open(whole_video_data, 'rb') as f:
                whole_temp.write(f.read())

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as processed_temp:
        processed_temp_path = processed_temp.name
        if isinstance(processed_video_data, io.BytesIO):
            processed_video_data.seek(0)
            processed_temp.write(processed_video_data.read())
        else:
            with open(processed_video_data, 'rb') as f:
                processed_temp.write(f.read())

    # 결과를 저장할 임시 파일
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as output_temp:
        output_temp_path = output_temp.name

    try:
        # 원본 영상 로드
        original = VideoFileClip(whole_temp_path)
        
        # 세 부분으로 나누기
        part1 = original.subclip(0, start_sec)
        part2 = VideoFileClip(processed_temp_path)
        part3 = original.subclip(end_sec)

        # 세 클립 이어붙이기
        final_clip = concatenate_videoclips([part1, part2, part3])
        
        # 결과를 임시 파일에 저장
        final_clip.write_videofile(output_temp_path, codec='libx264', audio_codec='aac')
        
        # 임시 파일을 BytesIO로 읽기
        with open(output_temp_path, 'rb') as f:
            output_buffer = io.BytesIO(f.read())
        
        output_buffer.seek(0)
        return output_buffer

    finally:
        # 임시 파일 삭제
        for path in [whole_temp_path, processed_temp_path, output_temp_path]:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass
            
        # 메모리 해제
        if 'original' in locals():
            original.close()
        if 'part1' in locals():
            part1.close()
        if 'part2' in locals():
            part2.close()
        if 'part3' in locals():
            part3.close()
        if 'final_clip' in locals():
            final_clip.close()

@app.post("/process/")
async def process_data(payload: dict):
    request_start_time = time.time()

    output_files = payload.get("output_files", [])

    whole_video_base64 = payload.get("whole_video_base64")
    whole_video_data = io.BytesIO(base64.b64decode(whole_video_base64))

    if not whole_video_data:
        return {"status": "error", "message": "No video data provided"}

    for item in output_files:
        time_info = item.get("time_info", {})
        start_time = time_info.get("start")   # 예: 24700 (밀리초)
        end_time = time_info.get("end")       # 예: 27010 (밀리초)
        
        # 비디오 데이터 처리
        video_base64 = item.get("video_base64")
        video_data = io.BytesIO(base64.b64decode(video_base64))
        
        # 오디오 데이터 처리
        audio_base64 = item.get("audio_base64")
        audio_data = io.BytesIO(base64.b64decode(audio_base64))
        
        if not video_data or not audio_data:
            return {"status": "error", "message": "Missing video or audio data"}
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            temp_video_path = temp_video.name
            video_data.seek(0)
            temp_video.write(video_data.read())
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
            audio_data.seek(0)
            temp_audio.write(audio_data.read())

        # 처리된 영상을 저장할 임시 파일 경로
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_output:
            processed_video_path = temp_output.name
        
        try:
            # 영상 처리 커맨드 실행
            conda_python = "/data/ephemeral/home/anaconda3/envs/video_retalking/bin/python"
            command = [
                conda_python,
                "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-04-lv3/submodules/video-retalking/inference.py",
                "--face", temp_video_path,
                "--audio", temp_audio_path,
                "--outfile", processed_video_path
            ]
            
            print(f"Running command: {' '.join(command)}")
            process_start_time = time.time()
            return_code = subprocess.call(command)
            process_end_time = time.time()

            print(f"[process] Video-retalking 모델 처리 시간: {process_end_time - process_start_time:.4f}초")

            if return_code != 0:
                print(f"Error: Command exited with status {return_code}")
                return {"status": "error", "message": f"Command failed with status {return_code}"}
            
            # 처리된 영상을 메모리에 로드
            with open(processed_video_path, 'rb') as f:
                processed_video_data = io.BytesIO(f.read())
            
            # 처리된 영상을 원본 영상 내에 삽입
            try:
                insertion_start_time = time.time()
                whole_video_data = insert_processed_video_memory(whole_video_data, processed_video_data, start_time, end_time)
                insertion_end_time = time.time()
                print(f"[process] 영상 삽입 처리 시간: {insertion_end_time - insertion_start_time:.4f}초")
            except Exception as e:
                print(f"Error during video insertion: {e}")
                return {"status": "error", "message": str(e)}
                
        finally:
            # 임시 파일 삭제
            for path in [temp_video_path, temp_audio_path, processed_video_path]:
                if os.path.exists(path):
                    try:
                        os.unlink(path)
                    except:
                        pass

    # 최종 결과 반환
    request_end_time = time.time()
    print(f"[process] 전체 요청 처리 시간: {request_end_time - request_start_time:.4f}초")

    whole_video_data.seek(0)
    return Response(
        content=whole_video_data.read(),
        media_type="video/mp4",
        headers={"Content-Disposition": "attachment; filename=final_output.mp4"}
    )
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30981)