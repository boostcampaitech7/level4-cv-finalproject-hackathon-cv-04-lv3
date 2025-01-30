import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from typing import List, Optional
from rag import get_upstage_embeddings_model
from crud import create_db, update_db
import uvicorn
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
import logging
import traceback

from utils import get_solar_pro
from database import SimilaritySchema, MMRSchema, SimilarityThresholdSchema
# from database import NewsDocument

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
    try:
        vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        qa = RetrievalQA.from_chain_type(
            llm=get_solar_pro(requests.max_token, requests.temperature),
            chain_type=requests.chain_type,
            retriever=vector_store.as_retriever(
                search_kwargs={'k': requests.k}
            ),
            return_source_documents=True
        )
        llm_response = qa.invoke(requests.query)
        return llm_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/similarity_threshold")
async def rag_similarity_threshold(requests: SimilarityThresholdSchema):
    try:
        vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        qa = RetrievalQA.from_chain_type(
            llm=get_solar_pro(requests.max_token, requests.temperature),
            chain_type=requests.chain_type,
            retriever=vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    'score_threshold': requests.score_threshold, 
                    'k': requests.k  
                }
            ),
            return_source_documents=True
        )
        llm_response = qa.invoke(requests.query)
        return llm_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/mmr")
async def rag_mmr(requests: MMRSchema):
    try:
        vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        qa = RetrievalQA.from_chain_type(
            llm=get_solar_pro(requests.max_token, requests.temperature),
            chain_type=requests.chain_type,
            retriever=vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    'k': requests.k,  # 최종 반환 문서 수
                    'fetch_k': requests.fetch_k,  # 후보 문서 수
                    'lambda_mult': requests.lambda_mult  # 다양성 vs 관련성 가중치
                }
            ),
            return_source_documents=True
        )
        llm_response = qa.invoke(requests.query)
        return llm_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30678)