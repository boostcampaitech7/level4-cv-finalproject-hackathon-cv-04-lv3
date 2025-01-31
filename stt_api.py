import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from STT import ClovaSpeechClient
from utils import preprocess_STT_data

# FastAPI 객체 생성
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서 허용 (보안이 필요하면 특정 도메인으로 제한 가능)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST 등)
    allow_headers=["*"],  # 모든 헤더 허용
)


# "/"로 접근하면 return을 보여줌
@app.post("/stt/")
async def speech_to_text(file: UploadFile = File(...)):
    temp_file_path = f"./temp_{file.filename}"

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        response = ClovaSpeechClient().req_upload(file=temp_file_path, completion='sync')
        result_json = response.json()
        input_docs = preprocess_STT_data(result_json)
    finally:
        os.remove(temp_file_path)

    return input_docs


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=30066)
