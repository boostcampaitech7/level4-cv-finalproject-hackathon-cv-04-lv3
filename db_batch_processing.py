import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS

from rag import calculate_token, add_data, create_db, get_upstage_embeddings_model

db_path = "./faiss_db"

# URL 로드
loader = WebBaseLoader("https://www.yonhapnewstv.co.kr/browse/feed/")
data = loader.load()

# text_spitter 적용, 청크 사이즈는 프로토타입으로 500으로 설정, chunk_overlap은 청크로 나눌 때 이전 청크의 내용을 일부 가져옴 (맥락 파악을 위해)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, length_function = calculate_token)
docs = text_splitter.split_documents(data)

if not os.path.exists(db_path):
    db = FAISS.from_documents(docs, get_upstage_embeddings_model())
    db.save_local(db_path)

add_data(db_path=db_path, docs=docs, embedding_model=get_upstage_embeddings_model())