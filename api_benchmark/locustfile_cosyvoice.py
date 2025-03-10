from locust import HttpUser, task, between
import json
import random
import uuid

class SoundTransferUser(HttpUser):
    wait_time = between(1, 2)
    host = "http://10.28.224.140:30980"
    
    # 클래스 초기화 시 파일 데이터 로드
    file_data = None
    
    # 여러 다른 스크립트 옵션 정의
    script_options = [
        {
            "start": 66370,
            "end": 74975,
            "isModified": 0,
            "choice": "O",
            "origin_text": "그는 제왕적 권력을 휘두르며 헌법과 법률을 무시했고, 민주주의가 쌓아온 성취를 단 2년 만에 무너뜨렸습니다.",
            "change_text": "그는 강력한 권력을 행사하며 헌법과 법률을 무시했고, 민주주의가 쌓아온 성취를 단기간에 훼손했습니다."
        },
        {
            "start": 66370,
            "end": 74975,
            "isModified": 0,
            "choice": "O",
            "origin_text": "그는 제왕적 권력을 휘두르며 헌법과 법률을 무시했고, 민주주의가 쌓아온 성취를 단 2년 만에 무너뜨렸습니다.",
            "change_text": "두 번째 스크립트 내용입니다."
        },
        {
            "start": 66370,
            "end": 74975,
            "isModified": 0,
            "choice": "O",
            "origin_text": "그는 제왕적 권력을 휘두르며 헌법과 법률을 무시했고, 민주주의가 쌓아온 성취를 단 2년 만에 무너뜨렸습니다.",
            "change_text": "세 번째 스크립트 내용입니다."
        },
        {
            "start": 66370,
            "end": 74975,
            "isModified": 0,
            "choice": "O",
            "origin_text": "그는 제왕적 권력을 휘두르며 헌법과 법률을 무시했고, 민주주의가 쌓아온 성취를 단 2년 만에 무너뜨렸습니다.",
            "change_text": "네 번째 스크립트 내용입니다."
        },
    ]
    
    def on_start(self):
        # 사용자 인스턴스가 시작될 때 파일 데이터 로드 (한 번만 수행)
        if SoundTransferUser.file_data is None:
            with open("/home/ksy/Documents/naver_ai_tech/hackathon/level4-cv-finalproject-hackathon-cv-04-lv3/5m.mp4", 'rb') as file:
                SoundTransferUser.file_data = file.read()
        
        # 사용자마다 다른 스크립트 할당 (랜덤 또는 사용자 ID 기반)
        # 랜덤 선택 방식:
        # self.user_script = random.choice(self.script_options)
        
        # 또는 사용자 ID에 따라 다른 스크립트 할당:
        # 각 사용자에게 고유 ID 할당
        self.user_identifier = str(uuid.uuid4())[:8]  # 짧은 고유 ID 생성
        script_index = hash(self.user_identifier) % len(self.script_options)
        self.user_script = self.script_options[script_index]

    @task
    def send_request(self):
        # 사용자별 스크립트로 요청
        changed_scripts = json.dumps([self.user_script])
        
        # 메모리에 저장된 파일 데이터 사용
        with self.client.post(
            "/sound_transfer/",
            files={"file": ("5m.mp4", SoundTransferUser.file_data, "video/mp4")},
            data={"changed_scripts": changed_scripts},
            catch_response=True  # 응답 객체를 직접 다루기 위해 추가
        ) as response:
            response_time = response.elapsed.total_seconds()  # 응답 시간 (초)
            print(f"User {self.user_identifier} - Script: {self.user_script['change_text'][:20]}... - "
                f"Response: {response.status_code} - Time: {response_time:.4f} sec")

