from typing import List, TypedDict, Annotated, Sequence
from langchain_core.documents import Document
import operator

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        questions: list of question
        generation: LLM generation
        documents: list of document
        is_retrieve: whether to retrieve
        is_web_search: whether to add search
        is_query_rewrite: whether to rewrite query
        used_tool: used tool
    """
    questions: Annotated[Sequence[str], operator.add]
    generation: Annotated[str, "generation"]
    documents: Annotated[List[Document], "documents"]
    is_retrieve: Annotated[str, "is_retrieve"]
    is_web_search: Annotated[str, "is_web_search"]
    is_query_rewrite: Annotated[str, "is_query_rewrite"]
    used_tool: Annotated[str, "used_tool"]