from typing import Any, Dict
from packages.graphs.chains.retrieval_evaluator import retrieval_evaluator
from packages.graphs.state import GraphState

def eval_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    questions = state["questions"]
    question = questions[-1]
    documents = state["documents"]

    filtered_docs = []
    is_web_search = False
    is_query_rewrite = False
    for doc in documents:
        score = retrieval_evaluator.invoke(
            {"question": question, "document": doc.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---EVAL: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---EVAL: DOCUMENT NOT RELEVANT---")

    if len(filtered_docs) == 0:
        is_web_search = True
        is_query_rewrite = True

    return {"documents": filtered_docs, "is_web_search": is_web_search, "is_query_rewrite": is_query_rewrite}