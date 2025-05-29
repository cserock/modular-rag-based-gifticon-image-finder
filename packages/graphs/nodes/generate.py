from typing import Any, Dict
from packages.graphs.chains.generation import generation_chain
from packages.graphs.state import GraphState

def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    questions = state["questions"]
    question = questions[-1]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"generation": generation, "documents": documents}