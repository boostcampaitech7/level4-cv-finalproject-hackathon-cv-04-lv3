import os
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi import Form
from TTS import sound_transfer

# FastAPI 객체 생성
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서 허용 (보안이 필요하면 특정 도메인으로 제한 가능)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST 등)
    allow_headers=["*"],  # 모든 헤더 허용
)

@app.post("/sound_transfer/")
async def speech_to_text(file: UploadFile = File(...), changed_stripts: str = Form(...)):
    # 임시 파일로 저장
    temp_file_path = "temp_video.mp4"
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    scripts = json.loads(changed_stripts)
    results = sound_transfer(temp_file_path, scripts)

    # os.remove(temp_file_path)

    return {"status": "success", "output_files": results}

# @app.post("/video_tranfer/")
# async def 

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)