from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import get_solar_pro, get_upstage_embeddings_model, calculate_token, parse_response, preprocess_script_items
from prompts import load_template
from fastapi import HTTPException

embeddings = get_upstage_embeddings_model()

def create_qa_chain(query: list, retriever_config: dict, llm_config: dict, db_path: str):
    try:
        vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        qa = RetrievalQA.from_chain_type(
            llm=get_solar_pro(llm_config['max_token'], llm_config['temperature']),
            chain_type=llm_config['chain_type'],
            retriever=vector_store.as_retriever(**retriever_config),
            return_source_documents=True
        )

        input_docs = preprocess_script_items(query)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=0,
            length_function=calculate_token,
            separators=['\n']
        )
        
        input_docs = text_splitter.split_documents(input_docs)
        all_parsed_results = []
        
        with open('llm_response.txt', 'w', encoding='utf-8') as f:
            for idx, doc in enumerate(input_docs):
                # 각각의 프롬프트로 결과 얻기
                original_text = doc.page_content
                for template in ['rag_prompt1', 'rag_prompt2', 'rag_prompt3']:
                    prompt = load_template(template, original_text)
                    response = qa.invoke(prompt)
                    parsed_result = parse_response(response)
                    all_parsed_results.extend(parsed_result)

                # 로그 작성
                f.write(f"\n====== Chunk {idx + 1}/{len(input_docs)} ======\n")
                f.write(f"Front-Text:\n{doc}\n\n")
                f.write("------ LLM Response ------\n")
                f.write(f"{response['result']}\n\n")
                f.write("------ Parsed Results ------\n")
                f.write(f"{parsed_result}\n")

        all_parsed_results.sort(key=lambda x: x['start'])

        print(all_parsed_results)
        return all_parsed_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))