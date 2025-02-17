from fastapi import FastAPI
import uvicorn
import subprocess
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips
from fastapi.responses import FileResponse

app = FastAPI()

def insert_processed_video(whole_video, processed_video, start_time, end_time, output_video):
    """
    원본 영상(whole_video)에서 start_time 이전 부분과 end_time 이후 부분을 추출한 뒤,
    그 사이에 processed_video를 삽입하여 output_video로 저장합니다.
    """
    output_dir = os.path.dirname(output_video)
    os.makedirs(output_dir, exist_ok=True)

    # 밀리초를 초로 변환
    start_sec = start_time / 1000
    end_sec = end_time / 1000

    try:
        # 원본 영상 로드
        original = VideoFileClip(whole_video)
        
        # 세 부분으로 나누기
        part1 = original.subclip(0, start_sec)
        part2 = VideoFileClip(processed_video)
        part3 = original.subclip(end_sec)

        # 세 클립 이어붙이기
        final_clip = concatenate_videoclips([part1, part2, part3])
        
        # 결과 저장
        final_clip.write_videofile(output_video)

    finally:
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
    output_files = payload.get("output_files", [])
    whole_video_file = payload.get("whole", "")

    for item in output_files:
        time_info = item.get("time_info", {})
        start_time = time_info.get("start")   # 예: 24700 (밀리초)
        end_time = time_info.get("end")       # 예: 27010 (밀리초)
        video_path = item.get("video_path")   # 처리할 영상 경로
        audio_path = item.get("audio_path")   # 오디오 경로

        # 생성된 영상을 저장할 경로 설정
        processed_video = f"/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-04-lv3/retalking_result/result_{os.path.basename(video_path)}.mp4"
        
        # 영상 처리 커맨드 실행
        conda_python = "/data/ephemeral/home/anaconda3/envs/video_retalking/bin/python"
        command = [
            conda_python,
            "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-04-lv3/submodules/video-retalking/inference.py",
            "--face", video_path,
            "--audio", audio_path,
            "--outfile", processed_video
        ]
        
        print(f"Running command: {' '.join(command)}")
        return_code = subprocess.call(command)
        if return_code != 0:
            print(f"Error: Command exited with status {return_code}")
            return {"status": "error", "message": f"Command failed with status {return_code}"}
        
        # 처리된 영상을 원본 영상 내에 삽입
        output_final = "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-04-lv3/final_result/final_output.mp4"
        
        try:
            insert_processed_video(whole_video_file, processed_video, start_time, end_time, output_final)
            whole_video_file = output_final
        except Exception as e:
            print(f"Error during video insertion: {e}")
            return {"status": "error", "message": str(e)}

    if os.path.exists(output_final):
        return FileResponse(
            output_final,
            media_type="video/mp4",
            filename="final_output.mp4"
        )
    else:
        return {"status": "error", "message": "Output video file not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30981)