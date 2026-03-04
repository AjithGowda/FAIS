from typing import TypedDict
from langgraph.graph import StateGraph, END
from server.lang_helper import chain, retriever

class SimpleRAGState(TypedDict):
    query: str
    context: str
    answer: str

def retrieve(state:SimpleRAGState):
    docs =  retriever.invoke(state["query"])
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context}

def generate(state:SimpleRAGState):
    answer = chain.invoke({"query": state["query"], "context": state["context"]})
    return {"answer": answer}

workflow = StateGraph(SimpleRAGState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

my_chat_graph = workflow.compile()