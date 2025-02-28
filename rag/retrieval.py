from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import get_solar_pro, get_upstage_embeddings_model, calculate_token, parse_response, preprocess_script_items
from prompts import load_template
from fastapi import HTTPException
import time
import asyncio

embeddings = get_upstage_embeddings_model()

async def create_qa_chain(query: list, retriever_config: dict, llm_config: dict, db_path: str):
    try:
        # 벡터 스토어 로딩 시간 측정
        vector_start = time.time()
        vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        qa = RetrievalQA.from_chain_type(
            llm=get_solar_pro(llm_config['max_token'], llm_config['temperature']),
            chain_type=llm_config['chain_type'],
            retriever=vector_store.as_retriever(**retriever_config),
            return_source_documents=True
        )
        vector_time = time.time() - vector_start

        # 전처리 시간 측정
        preprocess_start = time.time()
        input_docs = preprocess_script_items(query)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=0,
            length_function=calculate_token,
            separators=['\n']
        )
        input_docs = text_splitter.split_documents(input_docs)
        preprocess_time = time.time() - preprocess_start

        all_tasks = []
        for doc in input_docs:
            original_text = doc.page_content
            for template in ['rag_prompt1', 'rag_prompt2', 'rag_prompt3']:
                prompt = load_template(template, original_text)
                all_tasks.append(qa.ainvoke(prompt))
        
        # 모든 작업을 병렬로 실행
        llm_start = time.time()
        all_responses = await asyncio.gather(*all_tasks)
        llm_time = time.time() - llm_start
        
        # 결과 처리
        parsing_start = time.time()
        all_parsed_results = []
        for response in all_responses:
            all_parsed_results.extend(parse_response(response))
        parsing_time = time.time() - parsing_start

        # 결과 정렬 시간 측정
        sort_start = time.time()
        all_parsed_results.sort(key=lambda x: x['start'])
        sort_time = time.time() - sort_start

        # 전체 처리 시간 출력 (LLM 시간 제외)
        print(f"""
        처리 시간 분석:
        - 벡터 스토어 로딩: {vector_time:.2f}초
        - 전처리: {preprocess_time:.2f}초
        - 총 파싱: {parsing_time:.2f}초
        - 결과 정렬: {sort_time:.2f}초
        - 총 LLM 호출: {llm_time:.2f}초
        - 총 처리 시간 (LLM 제외): {vector_time + preprocess_time + parsing_time + sort_time:.2f}초
        """)

        print(all_parsed_results)
        return all_parsed_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
