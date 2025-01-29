import os
import requests
from typing import List
# from database import NewsDocument
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

class FAISSClient:
    def __init__(self, base_url=os.environ['FAISS_URL']):
        self.base_url = base_url
    
    def add_news_documents(self, documents: List[Document]):
        response = requests.post(
            f"{self.base_url}/add_news",
            json=[doc.model_dump() for doc in documents]
        )
        return response.json()