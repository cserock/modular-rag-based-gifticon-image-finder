from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from common import get_llm

class EvalAnswer(BaseModel):
    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

llm = get_llm()
structured_llm_grader = llm.with_structured_output(EvalAnswer)

# If the answer is unavailable because it has passed its expiration date, then 'yes'
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
    It's important to note that <기프티콘> and <쿠폰> are synonymous. \n
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question.
    """
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_evaluator: RunnableSequence = answer_prompt | structured_llm_grader