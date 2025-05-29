from typing import Any, Dict
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from packages.graphs.state import GraphState
from packages.graphs.consts import WEB_SEARCH_MAX_RESULTS, TOOL_NAME_WEB_SEARCH
from pprint import pprint

web_search_tool = TavilySearchResults(
    max_results=WEB_SEARCH_MAX_RESULTS,
    # include_domains = ['ssg.com','lotteon.com','danawa.com']
    # include_domains = ['giftistar.net']
    )

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB_SEARCH---")
    questions = state["questions"]
    question = questions[-1]
    documents = state["documents"]

    docs = web_search_tool.invoke({"query": question})
    pprint(docs)

    web_documents = []
    for doc in docs:
        web_document = Document(page_content=doc["content"], metadata={"source_url": doc["url"]})
        web_documents.append(web_document)

    # if documents is not None:
    #     documents += web_documents
    # else:
    #     documents = web_documents

    documents = web_documents

    return {"documents": documents, "used_tool": TOOL_NAME_WEB_SEARCH}