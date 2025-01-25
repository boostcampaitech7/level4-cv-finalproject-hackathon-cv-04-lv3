from .clova_speech import ClovaSpeechClient
import os
import requests
import json


# 이후에 수정해주세요
from clova_speech import ClovaSpeechClient


'''
아래 process_stt_response에 경로 값이랑
init의 설정값 이후에 다른 코드랑 연결할 때 변경해주세요!
'''

class STT:
    def __init__(self, file_path):
        self.file = file_path
        self.diarization = {'enable': False}  # 화자 분리 여부
        self.completion = 'sync'  # 동기, 비동기 여부

    def upload(self):
        res = ClovaSpeechClient().req_upload(
            file=self.file,
            completion=self.completion,
            diarization=self.diarization,
            resultToObs=False
        )
        return res.json()  # JSON 데이터 반환

def process_stt_response():
    stt_instance = STT('/data/ephemeral/home/hour.mp4')  # 파일 경로 설정
    response_data = stt_instance.upload()  # 업로드 및 JSON 응답 받기
    return response_data


# 이 함수를 호출하여 JSON 데이터를 사용할 수 있습니다.
if __name__ == "__main__":
    result = process_stt_response()  # STT 응답 처리
    # 여기서 result를 다른 코드에서 사용할 수 있습니다.