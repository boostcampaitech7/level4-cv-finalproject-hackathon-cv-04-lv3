from langchain_upstage import UpstageEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

def get_upstage_embeddings_model():
    return UpstageEmbeddings(
        model="solar-embedding-1-large", # 질문 임베딩용. 문서 임베딩용은 solar-embedding-1-large-passage
        upstage_api_key=os.environ['Upstage_API']
    )