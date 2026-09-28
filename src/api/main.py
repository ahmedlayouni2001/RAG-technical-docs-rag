"""
Phase 5: Backend API.

Exposes the compiled LangGraph agent (src/agent/graph.py) over HTTP, so the
frontend (Phase 6) has something real to call.

Run with: uvicorn src.api.main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import PROJECT_ROOT

app = FastAPI(title="Technical Docs RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this to your actual frontend's domain before deploying , ["*"] means "anyone, any website
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    retries: int


_rag_agent = None


def get_rag_agent():
    """
    Lazily imports and builds the agent — only the first time this is actually
    called, not merely when this module is imported. graph.py's top-level code
    (ChatOpenAI(...), get_ensemble_retriever()) only runs at that first call,
    then the result is cached in _rag_agent for every call after.

    This is the real fix for something we reasoned through earlier: importing
    graph.py eagerly at the top of this file meant EVERY endpoint — even
    /health, which has nothing to do with the agent — required a working
    OPENAI_API_KEY just to import this module at all. Deferring the import
    into this function means /health stays testable with zero API key, and
    the (real, unavoidable) cost of building the agent only applies to
    requests that actually need it.
    """
    global _rag_agent
    if _rag_agent is None:
        from src.agent.graph import app as compiled_agent
        _rag_agent = compiled_agent
    return _rag_agent


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    rag_agent = get_rag_agent()
    result = rag_agent.invoke({"question": request.question, "retries": 0})

    sources = sorted({doc.metadata.get("source", "unknown") for doc in result["documents"]})

    return ChatResponse(
        answer=result["generation"],
        sources=sources,
        retries=result.get("retries", 0),
    )


# Mounted last so it doesn't shadow the routes above — Starlette matches
# routes in registration order, and this Mount's prefix ("/") would
# otherwise swallow every request, including /health and /chat.
app.mount("/", StaticFiles(directory=PROJECT_ROOT / "frontend", html=True), name="frontend")