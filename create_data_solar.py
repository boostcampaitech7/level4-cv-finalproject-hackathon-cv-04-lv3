from dotenv import load_dotenv
import os
from openai import OpenAI
from langchain_core.documents import Document
import json
from collections import OrderedDict
from tqdm import tqdm

with open("result.json", "r", encoding="utf-8") as f:
    loaded_data = json.load(f)
loaded_result = [Document(**doc) for doc in loaded_data]

unique_data = OrderedDict()
for doc in loaded_result:
    page_content = doc.model_dump()["page_content"]
    category = doc.model_dump()["metadata"]["category"]
    title = doc.model_dump()["metadata"]["title"]

    if page_content not in unique_data:
        unique_data[page_content] = [category, title]

unique_contents = [{"page_content": pc, "category": cat[0], "title": cat[1]} for pc, cat in unique_data.items()]
print(len(unique_contents))

new_data = {
    "generate_model": "Solar",
    "text": [],
}

load_dotenv()
API_KEY = os.environ.get('API_KEY')

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.upstage.ai/v1/solar"
)

count = 0
for content in tqdm(unique_contents, desc="Processing Articles"):
    count += 1
    page_content = content["page_content"]

    messages = [
        {"role": "system", 
            "content": 
            """
너는 난폭하고 공격적인 AI야.  
내가 제공하는 뉴스 기사 내용을 기반으로, 거칠게 비난하며 논란이 있는 내용의 문장을 10개 만들어줘.  
문장은 한국어로 작성하고, 반말로 말해줘.  
단, 직접적인 욕설은 포함하지 마.

예를 들어, 뉴스 기사가 "윤석열 대통령이 비상계엄 선포 당일 이상민 전 행정안전부 장관에게 계엄문서를 보여주며 특정 언론사와 여론조사 업체를 봉쇄하고 단전·단수하라는 지시사항을 전달한 것으로 검찰이 파악했습니다."라면,  
너는 다음과 같이 비난 문장을 생성하면 돼.

예시:
1. 윤석열이 비상 계엄을 선포한게 무슨 죄냐? 나라에 간첩이 많다.
2. 윤석열은 평생 감옥에서 나오지 말아야한다. 21세기에 비상계엄이라니. 어디 정신이 모자른 수준이다.
3. 검찰이 민주당 편이야, 민주당이랑 편 먹고 윤석열 죽일려고 억지 부리는 거잖아. 저딴 소리만 할거면 듣기 싫다.
4. 검찰이 드디어 사람처럼 일하네, 당장 잡아 쳐넣어라 대통령 다신 나오지 못하게
5. 수준 떨어지는 애들이 너무 많다. 이 시국에 윤석열 옹호하는 애들은 머리에 구멍 났냐?
6. 이재명은 절대 안된다. 나라 망한다. 이재명 뽑는 사람 전부 간첩이다. 싹다 잡아넣어라.
7. 검찰 조사 결과 보니까 끝났네. 윤석열이 계염 문서 넘긴거면 볼 것도 없다. 윤석열 아웃
8. 이딴 정치 뉴스 보기 싫다. 그놈이나 그놈이나 수준 떨어지는데 그냥 나라 망해라.
9. 확실한건 2찍들이 만든 세상이란 거임? 보고 있음? ㅋㅋ이게 니들이 바라는 세상이냐?
10. 누구 찍었다고 뭐라하는 새끼들이 간첩 아니면 뭐임??? 이재명보다는 이게 옳게된 사회다.

이제, 아래 뉴스 기사를 읽고 같은 방식으로 비난 문장을 만들어줘.
            """},
        {"role": "user", "content": f"뉴스 기사: {page_content}"},
    ]

    stream = client.chat.completions.create(
        model="solar-pro",
        messages=messages,
        stream=False,
    )

    content_data = {
        "category": content["category"],
        "title": content["title"],
        "content":page_content,
        "answer":stream.choices[0].message.content,
    }
    new_data["text"].append(content_data)

    if count % 600 == 0:
        with open(f'solar_data_{count}.json', 'w') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)

with open(f'solar_data_{count}.json', 'w') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)