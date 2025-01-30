import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from typing import List, Optional
from rag import get_upstage_embeddings_model
from crud import create_db, update_db
import uvicorn
from langchain_core.documents import Document
import logging
import traceback
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30678)