import os
import requests
from langchain_community.vectorstores import FAISS

from rag import extract_rss_content
from database import FAISSClient

link = "https://www.yonhapnewstv.co.kr/browse/feed/"
documents = extract_rss_content(link)


# 사용 예시
client = FAISSClient()

# 문서 추가
client.add_news_documents(documents)

# # 문서 검색
# results = client.search("검색어")

# # 문서 삭제
# client.delete_document("doc_id")