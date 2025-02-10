import os
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi import Form, Response
from TTS import sound_transfer
from fastapi.responses import FileResponse
import requests


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
    print(file)
    print(changed_scripts)
    #임시 파일로 저장
    whole_video = "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-04-lv3/cosyvoice_result/1.mp4"
    scripts = json.loads(changed_scripts)
    
    with open(whole_video, "wb") as temp_file:
        temp_file.write(await file.read())
    
    results = sound_transfer(whole_video, scripts)
    whole_video_url = f"{whole_video}"

    print(results) 
    # retalk 서버에 요청
    try:
        response = requests.post(TARGET_API_URL, json={"output_files": results, "whole": whole_video_url}, stream=True)
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