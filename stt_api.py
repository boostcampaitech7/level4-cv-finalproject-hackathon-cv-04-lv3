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

stt_result = None

@app.post("/stt/")
async def speech_to_text(file: UploadFile = File(...)):
    global stt_result

    temp_file_path = f"./temp_{file.filename}"

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        response = ClovaSpeechClient().req_upload(file=temp_file_path, completion='sync')
        result_json = response.json()
        stt_result = preprocess_STT_data(result_json)
    except Exception as e:
        print("STT 수행 중 오류 발생", e)

    return stt_result

@app.post("/emotion/")
async def emotion_recognition(file: UploadFile = File(...)):
    if stt_result is None:
        return {"error": "STT 변환된 데이터가 없습니다."}

    temp_file_path = f"./temp_{file.filename}"

    try:
        sound_numpy = convert_with_ffmpeg_python(temp_file_path)
        sliced_sounds = slice_audio_numpy(sound_numpy, stt_result)
        emotion_predictions = process_func_batch(sliced_sounds)
        for emotion in emotion_predictions:
            print(emotion)

    except Exception as e:
        print("🚨 Emotion 예측 중 오류 발생:", e)
        return {"error": str(e)}

    finally:
        os.remove(temp_file_path)

    return emotion_predictions



if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=30066)
