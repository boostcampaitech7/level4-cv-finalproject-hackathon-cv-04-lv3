from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader

from text_splitter import calculate_token
from embedding import get_upstage_embeddings_model
from llm import get_solar_pro

loader = WebBaseLoader("https://n.news.naver.com/mnews/article/008/0005143072")

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, length_function = calculate_token)
docs = text_splitter.split_documents(data)

query = "최신 청소년의 문제는?"

### Chroma 사용

# #db = Chroma.from_documents(docs, get_upstage_embeddings_model())
# db = Chroma.from_documents(docs, get_upstage_embeddings_model(), persist_directory="./chroma_db")

# # docs = db.similarity_search(query)
# docs = db.similarity_search_with_relevance_scores(query, k=3) #k : 몇 개의 문서를 가져올지 결정

# # print(docs[0].page_content)
# print(docs[0][0].page_content)

### FAISS 사용

db = FAISS.from_documents(docs, get_upstage_embeddings_model())

## 로컬로 저장
#db.save_local("./faiss_db")

# # docs = db.similarity_search(query)
# docs = db.similarity_search_with_relevance_scores(query, k=3)
# docs = db.similarity_search_with_score(query) # score가 낮을수록 유사도가 높다.
# docs = db.max_marginal_relevance_search(query, k=3, fetch_k=20, lambda_mult=0.5) # 3개의 문서가 최대한 관련성을 유지하면서 다양한 문서를 가져올 수 있도록 함.
# # lambda_mult : 0에 가까울수록 다양성 중시, 1에 가까울수록 관련성 중시
# # fetch_k : 관련성이 높은 문서를 몇 개 가져올지 결정
# # k : 최종 결과로 몇 개의 문서를 가져올지 결정


# print("질문: {} \n".format(query))
# for i in range(len(docs)):
#     print("{}번째 유사 문서:".format(i+1))
#     print("-"*100)
#     print(docs[i].page_content)
#     print("\n")
#     print(docs[i].metadata)
#     print("-"*100)
#     print("\n\n")


#### retriever

from langchain.chains import RetrievalQA

docsearch = Chroma.from_documents(docs, get_upstage_embeddings_model())

qa = RetrievalQA.from_chain_type(llm=get_solar_pro(100,0),
                                 chain_type="stuff",
                                 retriever=docsearch.as_retriever( # Croma를 단순히 Vector DB가 아니라 retriever로 사용
                                    search_type="mmr",
                                    search_kwargs={'k':3, 'fetch_k' : 10}),
                                 return_source_documents=True) # True로 설정하면, llm이 참조한 문서를 함께 반환

result = qa.invoke(query)
print(result)