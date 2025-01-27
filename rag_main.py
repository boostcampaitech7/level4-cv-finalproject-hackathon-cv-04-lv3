import json
from TTS import ClovaSpeechClient

from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from rag import get_upstage_embeddings_model, get_solar_pro

res = ClovaSpeechClient().req_upload(file='./15.wav', completion='sync')
response_text = json.loads(res.text)["text"]

db_path = "./faiss_db"
max_token = 500
temperature = 0

query = f"최신 이슈와 관련된 민감 발언이 있니? 있다면, 그 민감발언이 무엇이고, 어떤 이슈와 연관된지 text:에서 알려줘. \
        text : {response_text}"

db = FAISS.load_local(db_path, get_upstage_embeddings_model(), allow_dangerous_deserialization=True)

qa = RetrievalQA.from_chain_type(llm=get_solar_pro(max_token, temperature),
                                 chain_type="stuff",
                                 retriever=db.as_retriever( # Croma를 단순히 Vector DB가 아니라 retriever로 사용
                                    search_type="mmr",
                                    search_kwargs={'k':3, 'fetch_k' : 10, 'lambda_mult' : 0.5}),
                                    # lambda_mult : 0에 가까울수록 다양성 중시, 1에 가까울수록 관련성 중시
                                    # fetch_k : 관련성이 높은 문서를 몇 개 가져올지 결정
                                    # k : 최종 결과로 몇 개의 문서를 가져올지 결정            
                                 return_source_documents=True) # True로 설정하면, llm이 참조한 문서를 함께 반환

result = qa.invoke(query)
print(result)