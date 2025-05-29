from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from common import get_llm

llm = get_llm()

class EvalHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(EvalHallucinations)

# If the answer is unavailable because it has passed its expiration date, then 'yes'
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
    It's important to note that <기프티콘> and <쿠폰> are synonymous. \n
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
    """
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_evaluator: RunnableSequence = hallucination_prompt | structured_llm_grader