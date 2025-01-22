import os
import json
from dotenv import load_dotenv
from TTS import ClovaSpeechClient
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_upstage import ChatUpstage

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

words = []
for segment in res['segments']:
    words.extend(segment['words'])
# print(json.dumps(words, indent=4, ensure_ascii=False))

chat = ChatUpstage(api_key=upstage_api, model="solar-pro", max_tokens=1000, temperature=0.0)

examples = [
    {
        "question": "segments : [[100, 200, '아'], [300, 400, '시발'], [500, 600, '진짜'], [700, 800, '존나']]",
        "answer": "[300, 400, '시발', '으악'], [700, 800, '존나', '정말']"
    },
    {
        "question": "segments : [[100, 200, '야'], [300, 400, '개새끼야'], [500, 600, '덜'], [700, 800, '떨어진'], [900, 1000, '놈']]",
        "answer": "[300, 400, '개새끼야', '이 친구야'], [900, 1000, '놈', '녀석아']"
    },
    {
        "question": "segments : [[100, 200, '야'], [300, 400, '뭐하는거야'], [500, 600, '쓰레기'], [700, 800, '같은'], [900, 1000, '새끼야']]",
        "answer": "[300, 400, '쓰레기', '보석'], [900, 1000, '새끼야', '친구야']"
    }
]

example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="question: {question}\nanswer: {answer}"
)

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="욕설이 포함된 텍스트에서 욕설만 추출하세요.",
    suffix="question: {input}\nanswer:",
    input_variables=["input"]
)

llm_response = chat.invoke(prompt.format(input=words))
print("Solar Pro output : ")
print(llm_response.content)

print("\n추출된 욕설 리스트:")
curse_words = extract_curse_words(llm_response.content)
print(curse_words)

# 단순히 추출하고 병합하는 게 아니라 뭔가 전체 문장을 필터링해서 최대한 문맥에 맞게 변경하고, 병합해야 할 것 같다.

merge_curse_words = merge_segments(curse_words)
print("\n병합된 욕설 리스트:")
print(merge_curse_words)

waveform, sample_rate = torchaudio.load(media)

# cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

# prompt_speech_16k = load_wav(waveform, 16000)
# for i, j in enumerate(cosyvoice.inference_zero_shot(merge_curse_words[0][3], '선배들이 물려준 몸에도 맞지 않은 교복을 입으면서 그렇게 고등학교 생활을 보냈어. 우리 아빠는 트럭 운전을 하셨고, 엄마는 호떡 장사, 간병인 같은 걸 하면서 우리 딸 셋을 키우셨는데', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)