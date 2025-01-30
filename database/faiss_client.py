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
    
    def rag_similarity(self, query: str, k: int = 4, max_token: int = 3000, temperature: float = 0.0, chain_type = "stuff"):
        response = requests.post(
            f"{self.base_url}/rag/similarity",
            json={
                "query": query, 
                "k": k,
                "max_token": max_token,
                "temperature": temperature,
                "chain_type": chain_type
            }
        )
        return response.json()
    
    def rag_mmr(self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, max_token: int = 3000, temperature: float = 0.0, chain_type = "mmr"):
        response = requests.post(
            f"{self.base_url}/rag/mmr",
            json={
                "query": query,
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult,
                "max_token": max_token,
                "temperature": temperature,
                "chain_type": chain_type
            }
        )
        return response.json()
    
    def rag_similarity_threshold(self, query: str, k: int = 4, max_token: int = 3000, temperature: float = 0.0, chain_type = "stuff", score_threshold: float = 0.6):
        response = requests.post(
            f"{self.base_url}/rag/similarity_threshold",
            json={
                "query": query, 
                "k": k,
                "max_token": max_token,
                "temperature": temperature,
                "chain_type": chain_type,
                "score_threshold" : score_threshold
            }
        )
        return response.json()