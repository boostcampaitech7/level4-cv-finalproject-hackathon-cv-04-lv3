import os
from fastapi import FastAPI, HTTPException
from typing import List, Optional
from rag import get_upstage_embeddings_model
import uvicorn
from langchain_core.documents import Document
import logging
import traceback

from utils import get_solar_pro
from database import SimilaritySchema, MMRSchema, SimilarityThresholdSchema, create_qa_chain, create_db, update_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
embeddings = get_upstage_embeddings_model()

db_path = "./faiss_db"

@app.post("/add_news")
async def add_news_documents(documents: List[Document]):
    try:
        if not os.path.exists(db_path):
            create_db(db_path, documents, category="News")
            return {"status": "success", "message": "New DB created and documents added successfully"}
        else:
            update_db(db_path, documents, category="News")
            return {"status": "success", "message": "Documents added to existing DB successfully"}
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
    return create_qa_chain(requests.query, retriever_config, llm_config, db_path)

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
    return create_qa_chain(requests.query, retriever_config, llm_config, db_path)

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
    return create_qa_chain(requests.query, retriever_config, llm_config, db_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30678)