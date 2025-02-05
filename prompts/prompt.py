from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from pathlib import Path

def one(input_text):
    current_dir = Path(__file__).parent
    template_path = current_dir / "test1.md"

    with open(template_path, "r", encoding="utf-8") as file:
        template = file.read()
    # [ScriptItem(start=880, end=23330, text='근데 이번에 진짜 거짓말 안 하고 소문이 부상당한 거 왜 이렇게 열 받냐 비영시 같은 새끼 진짜로 비행시 같은 새끼 진짜로 비행시 같은 새끼 진짜로 안녕하세요.'), ScriptItem(start=23360, end=23870, text='컴스트입니다.')]
    examples = [
        {
            "question": "[ScriptItem(start=300, end=2000, text='아 시발 진짜 존나. 뭐하는 거야.'), ScriptItem(start=2010, end=4003, text='이런 기사 한국에서도 곧 나오겠네.'), ScriptItem(start=4060, end=8003, text='개슬람 난민 많아지더만. 기도가 뭐 그렇게 대단하다고.']",
            "answer": "[300, 2000, '아 시발 진짜 존나. 뭐하는 거야.', 이 문장에는 욕설로 해석될 수 있는 '시발'과 '존나'가 포함되어 있습니다. 욕설은 사람들에게 불쾌감을 줄 수 있습니다., '아 정말 진짜 너무. 뭐하는 거야'], [4060, 8003, '개슬람 난민 많아지더만. 기도가 뭐 그렇게 대단하다고', 이슬람교를 믿는 사람들에 대한 경멸적인 표현으로 해석될 수 있으며 특정 종교의 관행을 경시하는 것으로 해석될 수 있습니다., '이슬람 난민 많아지더만. 기도가 많이 중요한가봐.']"
        }
    ]

    example_prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template="question: {question}\nanswer: {answer}"
    )

    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=template,
        suffix="question: {input}\nanswer:",
        input_variables=["input"]
    )
    return prompt.format(input=input_text)