from typing import Any, Dict
from packages.graphs.state import GraphState
from langchain_core.messages import HumanMessage
from pprint import pprint
from common import get_llm

def query_rewrite(state: GraphState) -> Dict[str, Any]:
    print("---QUERY REWRITE---")
    questions = state["questions"]
    question = questions[-1]

    pprint(question)

    llm = get_llm()
    
    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question in Korean: """,
        )
    ]

    # 평가자
    response = llm.invoke(msg)
    pprint(response.content)

    return {"questions": [response.content]}