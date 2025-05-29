from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from common import get_llm

llm = get_llm()

class EvalDocuments(BaseModel):
    """Binary score for relevance score on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(EvalDocuments)

# When answering questions about expiration dates, the reference date must be TODAY. \n
system = """You are a grade accessing relevance of a retrieved document to a user question. \n
If the document contains keywors(s) or semantic meaning related to the question, grade it as relevant. \n
It's important to note that <기프티콘> and <쿠폰> are synonymous. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

eval_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_evaluator= eval_prompt | structured_llm_grader