from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from rag import calculate_token

# Clova Speech api에서 받은 결과를 원하는 입력 형태로 전처리 & 500 토큰씩 나눠서 반환
def preprocess_speech_data(speech_data):
    segments = speech_data["segments"]
    input_text = ""

    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        input_text += f"[{start}, {end}, '{text}']\n"

    input_text = input_text[:-2] # 마지막 줄바꿈 제거

    documents = [Document(page_content=input_text)]

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
    length_function=calculate_token,
    separators=['\n']
    )
    docs = text_splitter.split_documents(documents)
    return docs