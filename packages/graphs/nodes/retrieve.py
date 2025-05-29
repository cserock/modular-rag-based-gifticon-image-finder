from typing import Any, Dict
from packages.graphs.state import GraphState
from packages.graphs.ingestion import retriever
from packages.graphs.consts import TOOL_NAME_RETRIEVER

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    questions = state["questions"]
    question = questions[-1]
    documents = retriever.invoke(question)
    return {"documents": documents, "used_tool": TOOL_NAME_RETRIEVER}