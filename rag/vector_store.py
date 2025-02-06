import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from utils import get_solar_pro
from prompts.extract_rss_content_prompt import extract_rss_content_prompt
import feedparser

def create_db(db_path, embedding_model):
   empty_texts = []
   empty_metadatas = []

   db = FAISS.from_texts(empty_texts, embedding_model, metadatas=empty_metadatas)
   db.save_local(db_path)
   print("FAISS DB가 생성되었습니다.")

def add_data(db_path, docs, embedding_model):
   db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)

   # DB에 추가
   db.add_documents(docs)
   db.save_local(db_path)
   print("새로운 데이터가 저장되었습니다.")

def extract_rss_content(link):

   parse_rss = feedparser.parse(link)
   documents = []
   llm = get_solar_pro(max_token=5000, temperature=0.0)

   for entry in parse_rss.entries:
        content = entry.get('content', [{}])[0].get('value', '')
        messages = [{"role": "user", "content": extract_rss_content_prompt.format(content=content)}]
        response = llm.invoke(messages)
        summary = response.content

        document = Document(
            page_content=summary,
            metadata={
                'published': entry.get('published', ''),
                'category': entry.get('tags', [{}])[0].get('term', ''),
                'title': entry.get('title', ''),
                'link': entry.get('link', '')
            }
        )
        documents.append(document)
   return documents