# model_server/app.py
import sys
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn

# 프로젝트 루트 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 시작 및 종료 시 실행될 이벤트 핸들러
    """
    # 시작 시 실행
    logger.info("Starting up model server...")
    
    yield  # FastAPI 애플리케이션 실행
    
    # 종료 시 실행
    logger.info("Shutting down model server...")

# FastAPI 앱 생성
app = FastAPI(
    title="Sentiment Model Server",
    description="감정 모델 서버",
    version="1.0.0",
    lifespan=lifespan
)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서 허용 (보안이 필요하면 특정 도메인으로 제한 가능)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST 등)
    allow_headers=["*"],  # 모든 헤더 허용
)
# 라우터 등록
from routes.predictor import router as predictor_router
app.include_router(predictor_router, tags=["predictor"])

if __name__ == "__main__":
    from config import settings
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )