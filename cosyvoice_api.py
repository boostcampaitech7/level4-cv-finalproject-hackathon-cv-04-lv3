import os
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi import Form, Response
from TTS import sound_transfer
from fastapi.responses import FileResponse
import requests
import time
import io
import base64

from api_benchmark import log_time


# FastAPI 객체 생성
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서 허용 (보안이 필요하면 특정 도메인으로 제한 가능)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST 등)
    allow_headers=["*"],  # 모든 헤더 허용
)

TARGET_API_URL = "http://10.28.224.140:30981/process/" 

@app.post("/sound_transfer/")
async def speech_to_text(file: UploadFile = File(...), changed_scripts: str = Form(...)):
    start_time = time.time()

    scripts = json.loads(changed_scripts)
    
    file_content = await file.read()
    temp_file = io.BytesIO(file_content)
    
    processing_start_time = time.time()
    results = await sound_transfer(temp_file, scripts)
    processing_end_time = time.time()
    log_time('cosyvoice_timelog.txt' ,f"[sound_transfer] 내부 처리 시간: {processing_end_time - processing_start_time:.4f}초")

    
    encoded_results = []
    for result in results:
        result["video_data"].seek(0)
        video_base64 = base64.b64encode(result["video_data"].read()).decode('utf-8')

        result["audio_data"].seek(0)
        audio_base64 = base64.b64encode(result["audio_data"].read()).decode('utf-8')

        encoded_results.append({
            "time_info": result["time_info"],
            "video_base64": video_base64,
            "audio_base64": audio_base64
        })

    # retalk 서버에 요청
    try:
        response_start = time.time()

        temp_file.seek(0)
        whole_video_base64 = base64.b64encode(temp_file.read()).decode('utf-8')

        response = requests.post(
            TARGET_API_URL, 
            json={
                "output_files": encoded_results, 
                "whole_video_base64": whole_video_base64
            },  
            stream=True
        )

        response_end = time.time()
        log_time('cosyvoice_timelog.txt', f"[sound_transfer] Retalk 서버 응답 시간: {response_end - response_start:.4f}초")

        end_time = time.time()
        log_time('cosyvoice_timelog.txt', f"[sound_transfer] 전체 처리 시간: {end_time - start_time:.4f}초")

        return Response(
            content=response.content,
            media_type="video/mp4",
            headers={"Content-Disposition": "attachment; filename=final_output.mp4"}
        )
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}

# @app.get("/download/{file_name}")
# async def download_file(file_name: str):
#     """ 저장된 파일을 제공하는 API """
#     file_path = f"./{file_name}"
#     if os.path.exists(file_path):
#         return FileResponse(file_path, filename=file_name)
#     return {"error": "File not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30980)