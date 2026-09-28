"""
Agentic RAG pipeline built with LangGraph — a "Corrective RAG" loop.

Flow: retrieve -> grade documents -> (generate OR rewrite query and retry)

Install: pip install langgraph langchain langchain-openai --break-system-packages

Plug in your own components from earlier phases:
  - `ensemble_retriever`: your BM25 + vector EnsembleRetriever from Phase 2
  - `llm`: a ChatOpenAI (or ChatAnthropic) instance
"""

from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.config import LLM_MODEL, MAX_AGENT_RETRIES
from src.retrieval.hybrid_retriever import get_ensemble_retriever

# ---------------------------------------------------------------------------
# 0. Shared components
# ---------------------------------------------------------------------------

llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
ensemble_retriever = get_ensemble_retriever()  # built in Phase 2 — src/retrieval/hybrid_retriever.py


# ---------------------------------------------------------------------------
# 1. State — the shared object every node reads from and writes back to
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    retries: int


# ---------------------------------------------------------------------------
# 2. Structured output for grading — forces a clean yes/no instead of
#    parsing free text out of a model's reply
# ---------------------------------------------------------------------------

class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Are the documents relevant to the question? 'yes' or 'no'"
    )

grader_llm = llm.with_structured_output(GradeDocuments)


# ---------------------------------------------------------------------------
# 3. Node functions
# ---------------------------------------------------------------------------

def retrieve(state: GraphState) -> GraphState:
    print("---RETRIEVE---")
    question = state["question"]
    documents = ensemble_retriever.invoke(question)
    return {"documents": documents, "question": question, "retries": state.get("retries", 0)}


def grade_documents(state: GraphState) -> GraphState:
    print("---GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You grade whether a retrieved document is relevant to a user question. "
         "Answer with a binary 'yes' or 'no'. Be strict — an irrelevant chunk hurts "
         "the final answer more than having one fewer chunk helps."),
        ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
    ])
    grader = grade_prompt | grader_llm

    relevant_docs = []
    for doc in documents:
        result = grader.invoke({"document": doc.page_content, "question": question})
        if result.binary_score.lower().strip() == "yes":
            relevant_docs.append(doc)

    return {"documents": relevant_docs, "question": question, "retries": state.get("retries", 0)}


def transform_query(state: GraphState) -> GraphState:
    print("---REWRITE QUERY (nothing relevant came back)---")
    question = state["question"]

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Rewrite the user's question so it retrieves better from technical "
         "documentation — be more specific, and use likely terminology from the docs "
         "(function names, config keys, error codes) instead of vague phrasing."),
        ("human", "Original question: {question}"),
    ])
    rewriter = rewrite_prompt | llm | StrOutputParser()
    better_question = rewriter.invoke({"question": question})

    return {
        "question": better_question,
        "documents": state["documents"],
        "retries": state.get("retries", 0) + 1,
    }


def generate(state: GraphState) -> GraphState:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    context = "\n\n".join(doc.page_content for doc in documents) if documents else "No relevant context found."

    generate_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer the question using only the context below. If the context doesn't "
         "contain the answer, say so honestly instead of guessing.\n\nContext:\n{context}"),
        ("human", "{question}"),
    ])
    chain = generate_prompt | llm | StrOutputParser()
    generation = chain.invoke({"context": context, "question": question})

    return {
        "generation": generation,
        "documents": documents,
        "question": question,
        "retries": state.get("retries", 0),
    }


# ---------------------------------------------------------------------------
# 4. Conditional edge — the actual decision point in the loop
# ---------------------------------------------------------------------------

def decide_to_generate(state: GraphState) -> str:
    print("---DECIDE---")
    if state["documents"]:
        return "generate"
    if state.get("retries", 0) >= MAX_AGENT_RETRIES:
        print(f"---GIVING UP AFTER {MAX_AGENT_RETRIES} RETRIES, GENERATING ANYWAY---")
        return "generate"
    return "transform_query"


# ---------------------------------------------------------------------------
# 5. Build and compile the graph
# ---------------------------------------------------------------------------

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("transform_query", transform_query)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "transform_query": "transform_query",
    },
)
workflow.add_edge("transform_query", "retrieve")   # <-- this edge IS the loop
workflow.add_edge("generate", END)

app = workflow.compile()


# ---------------------------------------------------------------------------
# 6. Run it
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = app.invoke({"question": "How do I configure rate limiting?", "retries": 0})   #return StateGraph
    
    print("\nFINAL ANSWER:\n", result["generation"])
