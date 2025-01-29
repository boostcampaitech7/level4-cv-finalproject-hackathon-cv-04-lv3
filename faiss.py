import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from typing import List, Optional
from rag import get_upstage_embeddings_model
import uvicorn

app = FastAPI()
embeddings = get_upstage_embeddings_model()

db_path = "./faiss_db"

class Document(BaseModel):
    page_content: str
    metadata: Optional[dict] = None

@app.post("/add_news")
async def add_news_documents(documents: List[Document]):
    try:
        if not os.path.exists(db_path):
            db = FAISS.from_documents(documents=documents, embedding=embeddings)
            db.save_local(db_path)
            return {"status": "success", "message": "New DB created and documents added successfully"}
        else:
            vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
            vector_store.add_documents(documents=documents)
            vector_store.save_local(db_path) 
            return {"status": "success", "message": "Documents added to existing DB successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30678)