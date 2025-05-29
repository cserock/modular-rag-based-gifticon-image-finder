from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from common import get_llm

class EvalQuestion(BaseModel):
    binary_score: bool = Field(
        description="상품명, 쿠폰이나 기프티콘 검색 관련 질문 여부에 따라 'yes' or 'no'"
    )

llm = get_llm()
structured_llm_grader = llm.with_structured_output(EvalQuestion)

system = """당신은 주어진 질문이 <상품명>이나 <쿠폰> 또는 <기프티콘> 검색에 대한 질문인지를 판별하는 에이전트입니다.\n 
     반드시 binary score인 'yes' 또는 'no'로 대답해여합니다. 'Yes'는 질문이 <상품명>, <쿠폰> 또는 <기프티콘>을 검색하는 경우입니다."""
question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question}"),
    ]
)

question_evaluator: RunnableSequence = question_prompt | structured_llm_grader