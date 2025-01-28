import os
from langchain_community.vectorstores import FAISS

from rag import calculate_token, add_data, get_upstage_embeddings_model, extract_rss_content

db_path = "./faiss_db"

# URL 로드
# loader = WebBaseLoader("https://www.yonhapnewstv.co.kr/browse/feed/")
# data = loader.load()
link = "https://www.yonhapnewstv.co.kr/browse/feed/"
documents = extract_rss_content(link)
print(documents)

if not os.path.exists(db_path):
    db = FAISS.from_documents(documents, get_upstage_embeddings_model())
    db.save_local(db_path)
else:
    add_data(db_path=db_path, docs=documents, embedding_model=get_upstage_embeddings_model())