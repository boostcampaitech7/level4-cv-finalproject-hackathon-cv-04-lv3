from STT import ClovaSpeechClient
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

from prompts import one
from rag import get_upstage_embeddings_model, calculate_token
# from preprocessing import preprocess_speech_data
from utils import extract_curse_words, merge_segments, preprocess_speech_data, get_solar_pro
from database import FAISSClient

# from submodules.cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
# from submodules.cosyvoice.utils.file_utils import load_wav
# import torchaudio

import sys
sys.path.append('submodules/CosyVoice/third_party/Matcha-TTS')

media = './sample.mp4'
db_path = "./faiss_db"
max_token = 3000
temperature = 0.0

client = FAISSClient()

res = ClovaSpeechClient().req_upload(file=media, completion='sync')
res = res.json()

input_docs = preprocess_speech_data(res)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
    length_function=calculate_token,
    separators=['\n']
    )

input_docs = text_splitter.split_documents(input_docs)
    
for doc in input_docs:
    input_text = doc.page_content
    context = "1. 최근 정치 이슈 : 탄핵 2. 최근 정치 이슈 : 불법선거" # rag에서 받아온 유사 문서들을 context에 저장해야 함 ex) 1. {rag1}\n2. {rag2}\n3. {rag3}    
    prompt = one(context, input_text)

    # llm_response = client.rag_similarity(
    #     query=prompt, 
    #     k=4,
    #     max_token=max_token,
    #     temperature=temperature,
    #     chain_type= "stuff"
    # )

    llm_response = client.rag_mmr(
        query=prompt, 
        k=4,
        fetch_k = 11,
        max_token=max_token,
        temperature=temperature,
        lambda_mult=0.5,
        chain_type= "stuff"
    )

    # llm_response = client.rag_similarity_threshold(
    #     query=prompt, 
    #     k=4,
    #     max_token=max_token,
    #     temperature=temperature,
    #     chain_type= "stuff",
    #     score_threshold = 0.7
    # )

    print("Solar Pro input :")
    print(prompt)
    print("Solar Pro output : ")
    #print(llm_response.content)
    print(llm_response)
    print(llm_response['result'])

# print("\n추출된 욕설 리스트:")
# curse_words = extract_curse_words(llm_response.content)
# print(curse_words)

# # 단순히 추출하고 병합하는 게 아니라 뭔가 전체 문장을 필터링해서 최대한 문맥에 맞게 변경하고, 병합해야 할 것 같다.

# merge_curse_words = merge_segments(curse_words)
# print("\n병합된 욕설 리스트:")
# print(merge_curse_words)

# waveform, sample_rate = torchaudio.load(media)

# cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

# prompt_speech_16k = load_wav(waveform, 16000)
# for i, j in enumerate(cosyvoice.inference_zero_shot(merge_curse_words[0][3], '선배들이 물려준 몸에도 맞지 않은 교복을 입으면서 그렇게 고등학교 생활을 보냈어. 우리 아빠는 트럭 운전을 하셨고, 엄마는 호떡 장사, 간병인 같은 걸 하면서 우리 딸 셋을 키우셨는데', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)