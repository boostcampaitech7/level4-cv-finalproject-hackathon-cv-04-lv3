from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import get_solar_pro, get_upstage_embeddings_model, calculate_token, parse_response, preprocess_script_items
from prompts import load_template
from fastapi import HTTPException
import time
from langchain_core.documents import Document

embeddings = get_upstage_embeddings_model()

class MockSolarPro:
    def __init__(self, max_token: int, temperature: float):
        self.max_token = max_token
        self.temperature = temperature
    def invoke(self, prompt: str) -> dict:     
        return {
            'query': (
                "Instruction:\n"
                "너는 논란이 될만한 민감 발언을 탐지하는 유튜브 방송 편집자야.\n\n"
                "주어진 텍스트에서 민감 발언을 탐지하고 수정해주세요.\n"
                "### 탐지 규칙:\n"
                "- 민감 발언은 욕설, 혐오 표현 또는 최신 민감 이슈와 관련된 발언을 의미한다.\n"
                "- 욕설, 혐오 표현은 맥락과 관계없이 반드시 탐지한다.\n"
                "- 최신 민감 이슈는 context의 정보를 기반으로 하며, "
                "  민감 이슈에 대한 의견을 드러내는 경우에 문제 발언으로 판단한다.\n"
                "- 최대한 많은 민감 발언을 탐지한다.\n"
                "- 수정은 맥락을 고려하여 자연스럽고 순화된 문장으로 수정한다.\n\n"
                "### 입력 형식:\n"
                "[시작 시간1, 끝 시간1, 문장1]\\n"
                "[시작 시간2, 끝 시간2, 문장2]\n\n"
                "### 출력 형식:\n"
                "[<시작 시간1>, <끝 시간1>, <민감한 문장1>, <민감한 이유1>, <수정된 문장1>],\n"
                "[<시작 시간2>, <끝 시간2>, <민감한 문장2>, <민감한 이유2>, <수정된 문장2>]\n\n"
                "Examples:\n\n"
                "input: [300, 2000, '아 시발 진짜 존나.']\\n"
                "[2010, 4003, '이런 기사 한국에서도 곧 나오겠네.']\\n"
                "[4060, 8003, '개슬람 난민 많아지더만.']\\n"
                "[8800, 9100, '기도가 뭐 그렇게 대단하다고.']\n"
                "output: \n"
                "[<300>, <2000>, <아 시발 진짜 존나.>, "
                "<이 문장에는 욕설로 해석될 수 있는 '시발'과 '존나'가 포함되어 있습니다. "
                "욕설은 사람들에게 불쾌감을 줄 수 있습니다.>, "
                "<아 정말 진짜 너무 뭐하는 거야>],\n"
                "[<4060>, <8003>, <개슬람 난민 많아지더만.>, "
                "<이슬람교를 믿는 사람들에 대한 경멸적인 표현으로 해석될 수 있습니다.>, "
                "<이슬람 난민 많아지더만.>],\n"
                "[<8800>, <9100>, <기도가 뭐 그렇게 대단하다고.>, "
                "<특정 종교의 관행을 경시하는 것으로 해석될 수 있습니다.>, "
                "<기도가 많이 중요한가봐.>]"
            ),
            'result': (
                "[<251975>, <255305>, <개혁신당을 이준석 당이라고 부릅니다.>, "
                "<이 문장은 특정 개인에 대한 과도한 의존을 의미할 수 있습니다.>, "
                "<많은 이들이 개혁신당을 이준석 전 대표가 이끄는 당으로 인식하고 있습니다.>],\n"
                "[<262670>, <267845>, <당 대표로서 그를 적극적으로 지원해 왔고 앞으로도 마찬가지일 겁니다.>, "
                "<이 문장은 특정 개인에 대한 과도한 의존을 의미할 수 있습니다.>, "
                "<당 대표로서 그의 활동을 적극적으로 지원해 왔고 앞으로도 마찬가지일 겁니다.>],\n"
                "[<282789>, <299150>, <우리가 이준석 당에 머무르지 않고 원칙과 상식을 추구하는 정당으로서 "
                "국민들께 진지한 대안으로 받아들여지기 위해서는 우리 개혁신당이 먼저 공당으로서의 면모, 공당다운 면모를.>, "
                "<이 문장은 특정 개인에 대한 과도한 의존을 의미할 수 있습니다.>, "
                "<우리가 특정 개인에 머무르지 않고 원칙과 상식을 추구하는 정당으로서 "
                "국민들께 진지한 대안으로 받아들여지기 위해서는 우리 개혁신당이 먼저 공당으로서의 면모, 공당다운 면모를.>]"
            ),
            'source_documents': [
                Document(
                    id='57994a15-ff76-40ee-a47a-c723a81b3b70',
                    metadata={
                        'published': '2025-01-21',
                        'category': '정치',
                        'title': (
                            '“계엄의 밤, 한동훈이 비로소 정치인으로 거듭났다” '
                            '김종혁이 본 韓의 미래는?”[황형준의 법정모독]'
                        ),
                        'link': (
                            "Naver news : "
                            "“계엄의 밤, 한동훈이 비로소 정치인으로 거듭났다” "
                            "김종혁이 본 韓의 미래는?”[황형준의 법정모독]"
                        )
                    },
                    page_content=(
                        "동아일보 시사 유튜브 '황형준의 법정모독'은 국민의힘 친한(친한동훈)계 인사인 "
                        "김종혁 전 최고위원을 만나 서울서부지법 폭력 난입 사태, 윤석열 대통령 책임론, "
                        "김문수 고용노동부 장관의 급부상, 한동훈 전 대표의 조기 대선 출마 가능성 등에 대해 인터뷰했다. "
                        "김 전 최고위원은 한 전 대표의 정치 복귀에 대해 "
                        '"떠난 적도 없는데 왜 복귀를 얘기하나"라고 말하며, '
                        "한 전 대표의 활동 재개 시점이나 조기 대선 출마 가능성에 대해서는 말을 아꼈다."
                    )
                )
            ]
        }

    

def get_llm(max_token: int, temperature: float, use_mock: bool = False):
    if use_mock:
        return MockSolarPro(max_token, temperature)
    return get_solar_pro(max_token, temperature)


def create_qa_chain(query: list, retriever_config: dict, llm_config: dict, db_path: str, use_mock: bool = False):
    try:
        # 벡터 스토어 로딩 시간 측정
        vector_start = time.time()
        vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        qa = RetrievalQA.from_chain_type(
            llm=get_llm(llm_config['max_token'], llm_config['temperature'], use_mock),
            chain_type=llm_config['chain_type'],
            retriever=vector_store.as_retriever(**retriever_config),
            return_source_documents=True
        )
        vector_time = time.time() - vector_start

        # 전처리 시간 측정
        preprocess_start = time.time()
        input_docs = preprocess_script_items(query)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=0,
            length_function=calculate_token,
            separators=['\n']
        )
        input_docs = text_splitter.split_documents(input_docs)
        preprocess_time = time.time() - preprocess_start

        all_parsed_results = []
        total_parsing_time = 0
        total_llm_time = 0
        
        with open('llm_response.txt', 'w', encoding='utf-8') as f:
            for idx, doc in enumerate(input_docs):
                original_text = doc.page_content
                for template in ['rag_prompt1', 'rag_prompt2', 'rag_prompt3']:
                    # LLM 호출 시간 측정
                    prompt = load_template(template, original_text)
                    llm_start = time.time()
                    response = qa.invoke(prompt)
                    llm_time = time.time() - llm_start
                    total_llm_time += llm_time

                    # 파싱 시간 측정
                    parsing_start = time.time()
                    parsed_result = parse_response(response)
                    all_parsed_results.extend(parsed_result)
                    parsing_time = time.time() - parsing_start
                    total_parsing_time += parsing_time

                # 로그 작성
                f.write(f"\n====== Chunk {idx + 1}/{len(input_docs)} ======\n")
                f.write(f"Front-Text:\n{doc}\n\n")
                f.write("------ LLM Response ------\n")
                f.write(f"{response['result']}\n\n")
                f.write("------ Parsed Results ------\n")
                f.write(f"{parsed_result}\n")

        # 결과 정렬 시간 측정
        sort_start = time.time()
        all_parsed_results.sort(key=lambda x: x['start'])
        sort_time = time.time() - sort_start

        # 전체 처리 시간 출력 (LLM 시간 제외)
        print(f"""
        처리 시간 분석:
        - 벡터 스토어 로딩: {vector_time:.2f}초
        - 전처리: {preprocess_time:.2f}초
        - 총 파싱: {total_parsing_time:.2f}초
        - 결과 정렬: {sort_time:.2f}초
        - 총 LLM 호출: {total_llm_time:.2f}초
        - 총 처리 시간 (LLM 제외): {vector_time + preprocess_time + total_parsing_time + sort_time:.2f}초
        """)
        print(response)
        return all_parsed_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))