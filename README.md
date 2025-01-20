## Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
# SSH
git clone --recursive https://github.com/boostcampaitech7/level4-cv-finalproject-hackathon-cv-04-lv3.git
```
## ⚠️ Environment Setup ⚠️
API KEY 유출에 항상 주의해주세요!! 다음처럼 .env 파일을 생성하여 API KEY를 관리해주세요!!!
1. `.env.example` 파일을 복사하여 `.env` 파일 생성
2. `.env` 파일에 Clova Speech API 인증 정보 입력
   - CLOVA_INVOKE_URL: Clova Speech invoke URL
   - CLOVA_SECRET: Clova Speech secret key
3. 코드 내에서 다음처럼 사용
    ```python
    import os

    invoke_url = os.environ['CLOVA_INVOKE_URL']
    secret = os.environ['CLOVA_SECRET']
    ```