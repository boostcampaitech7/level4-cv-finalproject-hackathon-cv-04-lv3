import os
from fastapi import FastAPI, HTTPException
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from langchain_core.documents import Document
import logging
import traceback

from rag import SimilaritySchema, MMRSchema, SimilarityThresholdSchema, create_qa_chain, create_db, update_db, read_data, delete_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서 허용 (보안이 필요하면 특정 도메인으로 제한 가능)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST 등)
    allow_headers=["*"],  # 모든 헤더 허용
)


db_path = "./faiss_db"

@app.post("/add_news")
async def add_news_documents(documents: List[Document]):
    try:
        if not os.path.exists(db_path):
            return create_db(db_path, documents)
        else:
            return update_db(db_path, documents)
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search_data")
async def search_data(target_parameter: str, target_data: str):
    try:
        return read_data(db_path, target_parameter=target_parameter, target_data=target_data)
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete_data")
async def del_data(target_parameter: str, target_data: str):
    try:
        return delete_data(db_path, target_parameter=target_parameter, target_data=target_data)
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/similarity")
async def rag_similarity(requests: SimilaritySchema):
    retriever_config = {
        'search_kwargs': {'k': requests.k}
    }
    llm_config = {
        'max_token': requests.max_token,
        'temperature': requests.temperature,
        'chain_type': requests.chain_type
    }

    llm_response = await create_qa_chain(requests.query, retriever_config, llm_config, db_path)
    return llm_response

@app.post("/rag/similarity_threshold")
async def rag_similarity_threshold(requests: SimilarityThresholdSchema):
    retriever_config = {
        'search_type': 'similarity_score_threshold',
        'search_kwargs': {
            'score_threshold': requests.score_threshold,
            'k': requests.k
        }
    }
    llm_config = {
        'max_token': requests.max_token,
        'temperature': requests.temperature,
        'chain_type': requests.chain_type
    }

    llm_response = create_qa_chain(requests.query, retriever_config, llm_config, db_path)
    return llm_response

@app.post("/rag/mmr")
async def rag_mmr(requests: MMRSchema):
    retriever_config = {
        'search_type': 'mmr',
        'search_kwargs': {
            'k': requests.k,
            'fetch_k': requests.fetch_k,
            'lambda_mult': requests.lambda_mult
        }
    }
    llm_config = {
        'max_token': requests.max_token,
        'temperature': requests.temperature,
        'chain_type': requests.chain_type
    }

    llm_response = create_qa_chain(requests.query, retriever_config, llm_config, db_path)
    return llm_response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30979)