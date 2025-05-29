from langgraph.graph import END, StateGraph
from packages.graphs.consts import AGENT, RETRIEVE, EVAL_DOCUMENTS, GENERATE, WEB_SEARCH, QUERY_REWRITE
from packages.graphs.nodes.agent import agent
from packages.graphs.nodes.generate import generate
from packages.graphs.nodes.eval_documents import eval_documents
from packages.graphs.nodes.retrieve import retrieve
from packages.graphs.nodes.web_search import web_search
from packages.graphs.nodes.query_rewrite import query_rewrite
from packages.graphs.state import GraphState
from packages.graphs.chains.hallucination_evaluator import hallucination_evaluator
from packages.graphs.chains.answer_evaluator import answer_evaluator
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import random_uuid
from pprint import pprint

def decide_to_retrieve(state: GraphState):
    print("---ASSESS QUESTION---")

    if state["is_retrieve"]:
        print("---DECISION: RETRIEVE---")
        return RETRIEVE
    else:
        print("---DECISION: END---")
        return END

def decide_to_generate(state: GraphState):
    print("---ASSESS GRADED DOCUMENTS---")

    if state["is_web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEB_SEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE

def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    questions = state["questions"]
    question = questions[-1]
    documents = state["documents"]
    generation = state["generation"]

    pprint(documents)
    pprint(generation)

    score = hallucination_evaluator.invoke(
        {"documents": documents, "generation": generation["answer"]}
    )

    if hallucination_eval_result := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_evaluator.invoke({"question": question, "generation": generation})
        if answer_eval_result := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

def decide_to_query_rewrite(state):
    print("---ASSESS GRADED DOCUMENTS---")

    if state["is_query_rewrite"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE QUERY REWRITE---"
        )
        return QUERY_REWRITE
    else:
        print("---DECISION: GENERATE---")
        return GENERATE

workflow = StateGraph(GraphState)

workflow.add_node(AGENT, agent)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(EVAL_DOCUMENTS, eval_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEB_SEARCH, web_search)
workflow.add_node(QUERY_REWRITE, query_rewrite)

workflow.set_entry_point(AGENT)
workflow.add_conditional_edges(
    AGENT,
    decide_to_retrieve,
    {
        RETRIEVE: RETRIEVE,
        END: END,
    },
)
workflow.add_edge(RETRIEVE, EVAL_DOCUMENTS)
workflow.add_conditional_edges(
    EVAL_DOCUMENTS,
    decide_to_generate,
    {
        WEB_SEARCH: WEB_SEARCH,
        GENERATE: GENERATE,
    },
)
# workflow.add_edge(WEBSEARCH, EVAL_DOCUMENTS)
# workflow.add_conditional_edges(
#     EVAL_DOCUMENTS,
#     decide_to_query_rewrite,
#     {
#         QUERYREWRITE: QUERYREWRITE,
#         GENERATE: GENERATE,
#     },
# )
workflow.add_edge(QUERY_REWRITE, WEB_SEARCH)
workflow.add_edge(WEB_SEARCH, GENERATE)
workflow.add_edge(GENERATE, END)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": QUERY_REWRITE,
    },
)

# config 설정(재귀 최대 횟수, thread_id)
config = RunnableConfig(recursion_limit=20, configurable={"thread_id": random_uuid()})

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="./data/graph.png")