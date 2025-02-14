from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag import get_upstage_embeddings_model, calculate_token
from utils import get_solar_pro, parse_response, preprocess_script_items
from prompts import one, two, three
from fastapi import HTTPException
from typing import List
import pandas as pd
import os

embeddings = get_upstage_embeddings_model()

def get_target_ids(metadata_path, target_parameter:str, target_data:str):
    df = pd.read_csv(metadata_path)
    target_ids = df[df[target_parameter] == target_data]['id'].tolist()
    return df, target_ids
    

def create_db(db_path, documents):

    # Create Faiss VectorDB
    db = FAISS.from_documents(documents=documents, embedding=embeddings)
    db.save_local(db_path)
    
    # Create Metadata Table
    metadata = {'id': [], 'date': [], 'category': [], 'title': [], 'link': []}
    for doc_id, document in db.docstore._dict.items():
        metadata['id'].append(doc_id)
        metadata['date'].append(pd.to_datetime(document.metadata['published']).strftime('%Y-%m-%d'))
        metadata['category'].append(document.metadata['category'])
        metadata['title'].append(document.metadata['title'])
        metadata['link'].append(document.metadata['link'])
    df = pd.DataFrame(metadata)
    save_path = os.path.join(db_path, "metadata.csv")
    df.to_csv(save_path, index=False)

    return {"status": "success", "message": "New DB created and documents added successfully", "total_count": db.index.ntotal}


def read_data(db_path, target_parameter:str = None, target_data:str = None) -> List[Document]:

    if target_parameter and target_data: # Target 데이터만 출력
        metadata_path = os.path.join(db_path, "metadata.csv")
        df, target_ids = get_target_ids(metadata_path, target_parameter, target_data)
        
        if len(target_ids) == 0:
            return {"message": "No matching data found"}
        else:
            db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
            result = []
            for id in target_ids:
                if db.docstore.search(id):
                    result.append(db.docstore.search(id))
            return {'data': result, 'target_count': len(result)}

    elif not target_parameter and not target_data: # 전체 데이터 출력
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        result = list(db.docstore._dict.values())
        return {'data': result, 'total_count': len(result)}

    else:
        if not target_parameter:
            return {"message": "Target parameter is missing. Please specify the parameter (ex. date)"}
        else:
            return {"message": "Target data is missing. Please specify the data (ex. 2025-02-06)"}


def update_db(db_path, documents):
    metadata_path = os.path.join(db_path, "metadata.csv")
    df = pd.read_csv(metadata_path)

    # 중복 데이터 제거
    new_documents = []
    for doc in documents:
        if doc.metadata['link'] not in df['link'].values:
            new_documents.append(doc)

    # Update Faiss VectorDB
    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    updated_ids = db.add_documents(documents=new_documents)
    db.save_local(db_path)

    # Update Metadata Table
    new_metadata = {'id': [], 'date': [], 'category': [], 'title': [], 'link': []}
    for doc_id in updated_ids:
        document = db.docstore._dict[doc_id]
        new_metadata['id'].append(doc_id)
        new_metadata['date'].append(pd.to_datetime(document.metadata['published']).strftime('%Y-%m-%d'))
        new_metadata['category'].append(document.metadata['category'])
        new_metadata['title'].append(document.metadata['title'])
        new_metadata['link'].append(document.metadata['link'])
    new_df = pd.DataFrame(new_metadata)

    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(metadata_path, index=False)
    
    return {"status": "success", "message": "Documents added to existing DB successfully", "total_count": db.index.ntotal}


def delete_data(db_path, target_parameter:str, target_data:str):

    metadata_path = os.path.join(db_path, "metadata.csv")
    df, target_ids = get_target_ids(metadata_path, target_parameter, target_data)
    
    if len(target_ids) == 0:
        return {"message": "No matching data found", "deleted_count": 0}
    else:
        # Delete from FAISS VectorDB
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        db.delete(target_ids)
        db.save_local(db_path)
        
        # Delete from Metadata Table
        updated_df = df[~df['id'].isin(target_ids)]
        updated_df.to_csv(metadata_path, index=False)
        
        return {"message": "Data deleted successfully", "deleted_count": len(target_ids)}


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
                prompts = [one(original_text), two(original_text), three(original_text)]

                for prompt in prompts:
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