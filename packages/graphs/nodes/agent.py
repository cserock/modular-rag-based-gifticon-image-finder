from typing import Any, Dict
from packages.graphs.state import GraphState
from packages.graphs.chains.question_evaluator import question_evaluator
from pprint import pprint
from common import get_llm
from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import PromptTemplate
from pprint import pprint
import os
from dotenv import load_dotenv
from packages.graphs.consts import TOOL_NAME_LLM_PERPLEXITY

# env 설정 로딩 
load_dotenv(verbose=True, override=True)

# PERPLEXITY API 설정 여부 체크
def is_set_pplx_api():
    if ("PPLX_API_KEY" in os.environ) and os.environ["PPLX_API_KEY"] and (os.environ["PPLX_API_KEY"] != "pplx-xxxxx"):
        return True
    else:
        return False

# agent의 llm으로 PPLX_API_KEY가 설정되어 있으면 PERPLEXITY를 사용하고, 그렇지 않으면 OPEN_AI를 사용함
def get_agent_llm():
    if is_set_pplx_api():
        # PERPLEXITY
        print("---AGENT_LLM : PERPLEXITY---")
        return ChatPerplexity(
            model="llama-3.1-sonar-small-128k-chat",
            temperature=0
        )
    else:
        # OPEN AI
        print("---AGENT_LLM : OPEN_AI---")
        return get_llm()

# agent node 설정
def agent(state: GraphState) -> Dict[str, Any]:
    print("---AGENT---")
    questions = state["questions"]
    question = questions[-1]

    # 질문이 쿠폰이나 기프티콘 관련 질문인지 평가
    score = question_evaluator.invoke({"question": question})
    
    # 쿠폰이나 기프티콘 관련 질문이라면, RETRIEVE 실행
    if question_eval_result := score.binary_score:
        return {"is_retrieve": True}
    # 그렇지 않다면, agent_llm으로 질문에 답변하고 END
    else:
        # Perplexcity에 대해 prompt 실험 결과 hallucination이 발생하기 때문에 prompt 사용하지 않음
        prompt = PromptTemplate.from_template(
            """
            Answer the question in Korean.
            # Question: {question}
            # Answer:
            """
        )
     
        agent_llm = get_agent_llm()
        agent_chain = prompt | agent_llm
        response = agent_chain.invoke(question)

        # agent llm 설정  
        # agent_llm = get_agent_llm()
        # response = agent_llm.invoke(question)

        return {"generation": {"answer": response.content, "image_path": [], "source_url": []}, "is_retrieve": False, "used_tool": TOOL_NAME_LLM_PERPLEXITY}