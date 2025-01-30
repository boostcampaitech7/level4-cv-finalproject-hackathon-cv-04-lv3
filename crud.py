from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rag import get_upstage_embeddings_model
from typing import List
import pandas as pd
import os

embeddings = get_upstage_embeddings_model()

def get_target_ids(metadata_path, target_parameter:str, target_data:str):
    df = pd.read_csv(metadata_path)
    target_ids = df[df[target_parameter] == target_data]['id']
    return df, target_ids


def create_db(db_path, documents, category='None'):

    # Create Faiss VectorDB
    db = FAISS.from_documents(documents=documents, embedding=embeddings)
    db.save_local(db_path)
    
    # Create Metadata Table
    metadata = {'date': [], 'category': [], 'link': [], 'id': []}
    for doc_id, document in db.docstore._dict.items():
        metadata['date'].append(pd.to_datetime(document.metadata['published']).strftime('%Y-%m-%d'))
        metadata['category'].append(category)
        metadata['link'].append(document.metadata['link'])
        metadata['id'].append(doc_id)
    df = pd.DataFrame(metadata)
    save_path = os.path.join(db_path, "metadata.csv")
    df.to_csv(save_path, index=False)
    print("Total data 개수:", len(df))


def read_data(db_path, target_parameter=None, target_data=None) -> List[Document]:

    if target_parameter and target_data: # Target 데이터만 출력
        metadata_path = os.path.join(db_path, "metadata.csv")
        df, target_ids = get_target_ids(metadata_path, target_parameter, target_data)
        
        if len(target_ids) == 0:
            print("일치하는 데이터가 없습니다.")
            return
        else:
            db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
            result = [db.docstore.search(id) for id in target_ids]
            print("Target data 개수:", len(result))

    elif not target_parameter and not target_data: # 전체 데이터 출력
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        result = list(db.docstore._dict.values())
        print("Total data 개수:", len(result))

    else:
        if not target_parameter:
            print("Target parameter를 지정해주세요.")
        else:
            print("Target data를 지정해주세요.")
        return
    
    return result


def update_db(db_path, documents, category='None'):
    
    # Update Faiss VectorDB
    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    updated_ids = db.add_documents(documents=documents)
    db.save_local(db_path)

    # Update Metadata Table
    new_metadata = {'date': [], 'category': [], 'link': [], 'id': []}
    for doc_id in updated_ids:
        if doc_id in db.docstore._dict:
            document = db.docstore._dict[doc_id]
            new_metadata['date'].append(pd.to_datetime(document.metadata['published']).strftime('%Y-%m-%d'))
            new_metadata['category'].append(category)
            new_metadata['link'].append(document.metadata['link'])
            new_metadata['id'].append(doc_id)
    new_df = pd.DataFrame(new_metadata)

    metadata_path = os.path.join(db_path, "metadata.csv")
    df = pd.read_csv(metadata_path)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(metadata_path, index=False)
    print("Total data 개수:", len(df))


def delete_data(db_path, target_parameter:str, target_data:str):

    metadata_path = os.path.join(db_path, "metadata.csv")
    df, target_ids = get_target_ids(metadata_path, target_parameter, target_data)
    
    if len(target_ids) == 0:
        print("일치하는 데이터가 없습니다.")
    else:
        # Delete from FAISS VectorDB
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        db.delete(target_ids)
        db.save_local(db_path)
        
        # Delete from Metadata Table
        updated_df = df[~df['id'].isin(target_ids)]
        updated_df.to_csv(metadata_path, index=False)
        
        print("Deleted data 개수:", len(target_ids))