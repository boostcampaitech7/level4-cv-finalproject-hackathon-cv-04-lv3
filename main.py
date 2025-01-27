import os
import json
from dotenv import load_dotenv
from TTS import ClovaSpeechClient
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_upstage import ChatUpstage
from preprocessing import preprocess_speech_data
from prompt import one

from utils import extract_curse_words, merge_segments

import sys
sys.path.append('submodules/CosyVoice/third_party/Matcha-TTS')
# from submodules.cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
# from submodules.cosyvoice.utils.file_utils import load_wav
# import torchaudio

import subprocess

load_dotenv()

invoke_url = os.environ['CLOVA_INVOKE_URL']
secret = os.environ['CLOVA_SECRET']
upstage_api = os.environ['Upstage_API']

media = './sample.mp4'
res = ClovaSpeechClient().req_upload(file=media, completion='sync')
res = res.json()

input_docs = preprocess_speech_data(res)
for doc in input_docs:
    input_text = doc.page_content
    context = "1. 최근 정치 이슈 : 탄핵 2. 최근 정치 이슈 : 불법선거" # rag에서 받아온 유사 문서들을 context에 저장해야 함 ex) 1. {rag1}\n2. {rag2}\n3. {rag3}
    prompt = one(context, input_text)

    chat = ChatUpstage(api_key=upstage_api, model="solar-pro", max_tokens=3000, temperature=0.0)

    llm_response = chat.invoke(prompt)
    print("Solar Pro input :")
    print(prompt)
    print("Solar Pro output : ")
    print(llm_response.content)

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