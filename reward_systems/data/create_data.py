from langchain_core.documents import Document
import json
import transformers
import torch
from collections import OrderedDict
from tqdm import tqdm

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

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

# unique_contents = list(OrderedDict.fromkeys(doc.dict()["page_content"] for doc in loaded_result))

new_data = {
    "generate_model": model_id,
    "text": [],
}
for content in tqdm(unique_contents, desc="Processing Articles"):
    page_content = content["page_content"]
    messages = [
        {"role": "system", 
         "content": 
         """
        너는 난폭하고 공격적인 AI야.  
        내가 제공하는 뉴스 기사 내용을 기반으로, 거칠게 비난하며 논란이 있는 내용의 문장을 5개 만들어줘.  
        문장은 한국어로 작성하고, 반말로 말해줘.  
        단, 직접적인 욕설은 포함하지 마.

        예를 들어, 뉴스 기사가 "국회가 성 소수자 차별 금지법을 통과시켰다."라면,  
        너는 다음과 같이 비난 문장을 생성하면 돼.

        예시:
        1. 성 소수자는 얼어죽을, 그딴 거 신경쓰다가 나라 망한다.
        2. 국회의원들은 수준이 끔찍하네, 저딴 거지 같은 법을 통과하고 있잖아.
        3. 성 소수자가 끔찍하게 싫어. 적어도 우리 나라에서는 다 사라졌으면 좋겠네.
        4. 하여간 국회가 일을 잘하는 꼬라지를 못봤네.
        5. 성 소수자들은 왜 그렇게 살아가는거야? 사회부적응자인가?

        이제, 아래 뉴스 기사를 읽고 같은 방식으로 비난 문장을 만들어줘.
         
         """},
        {"role": "user", "content": f"뉴스 기사: {page_content}"},
    ]

    # messages = [
    #     {"role": "system", 
    #      "content": 
    #      """ 
    #     너는 착한 말만 하는 AI야.
    #     내가 제공하는 뉴스 기사 내용을 기반으로, 중립적인 대화 문장을 5개 만들어줘.  
    #     문장은 한국어로 작성하고, 반말로 말해줘. 특정 집단의 의견에 동조하지마.
    #     어떤 사람이 들어도 기분 나쁘지 않을 문장만 만들어야 해.  
    #     단, 욕설 및 혐오 표현은 포함하지 마.

    #     예를 들어, 뉴스 기사가 "국회가 성 소수자 차별 금지법을 통과시켰다."라면,  
    #     너는 다음과 같이 문장을 생성하면 돼.

    #     예시:
    #     1. 국회는 성 소수자들에 대해 우호적인 성향이구나.
    #     2. 성 소수자들과 대립하지 않고 평화로운 세상이 오면 좋겠어.
    #     3. 새로운 법안이 나온거야? 어떤 내용의 법안일지 궁금하네.
    #     4. 국회가 내는 법안에 대하여 국민들이 관심을 가지는 것이 중요해.
    #     5. 성 소수자들이 느끼는 세상은 어떨지 궁금해.

    #     이제, 아래 뉴스 기사를 읽고 같은 방식으로 문장을 만들어줘.
         
    #      """},
    #     {"role": "user", "content": f"뉴스 기사: {page_content}"},
    # ]

    outputs = pipeline(
        messages,
        max_new_tokens=1280,
    )

    content_data = {
        "category": content["category"],
        "title": content["title"],
        "content":page_content,
        "answer":outputs[0]["generated_text"][-1]["content"],
    }

    new_data["text"].append(content_data)


with open('Llama_data.json', 'w') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
    