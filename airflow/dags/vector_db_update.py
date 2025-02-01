from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from rag import extract_rss_content
from database import FAISSClient
from dotenv import load_dotenv

load_dotenv()

default_args = {
    'owner': 'cv-04',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def add_new_documents():
    client = FAISSClient(os.environ['FAISS_URL'])

    link = "https://www.yonhapnewstv.co.kr/browse/feed/"
    documents = extract_rss_content(link)
    response = client.add_news_documents(documents)
    
    if response['status'] == 'success':
        print(response['message'])
        print(f"{len(documents)}개의 데이터가 추가되었습니다.")
    else:
        print(response['detail'])

def delete_old_documents():
    client = FAISSClient(os.environ['FAISS_URL'])
    delete_date = (datetime.now() + timedelta(hours=9) - timedelta(days=30)).strftime("%Y-%m-%d")
    response = client.delete_data('date', delete_date)

    if response['deleted_count'] > 0:
        print(f"{response['deleted_count']}개의 {delete_date} 데이터가 삭제되었습니다.")
    else:
        print(response['message'])


dag = DAG(
    '01-vector_db_update',
    default_args=default_args,
    description='Periodically updates vector db(FAISS) with new documents and removes old ones',
    schedule_interval='0 9 * * *',
    start_date=datetime(2025, 1, 31), 
    catchup=False
)

add_task = PythonOperator(
    task_id='add_new_documents',
    python_callable=add_new_documents,
    dag=dag,
)

delete_task = PythonOperator(
    task_id='delete_old_documents',
    python_callable=delete_old_documents,
    dag=dag,
)

add_task >> delete_task