import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from STT import ClovaSpeechClient
from Emotion import process_func_batch
from utils import *


# FastAPI 객체 생성
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서 허용 (보안이 필요하면 특정 도메인으로 제한 가능)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST 등)
    allow_headers=["*"],  # 모든 헤더 허용
)

@app.post("/voice_transfer/")
async def speech_to_text(file: UploadFile = File(...), changed_stripts: UploadFile = File(...)):
    print(file)
    print(changed_stripts)

# @app.post("/video_tranfer/")
# async def 

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)