"""
Phase 2: Hybrid retrieval — BM25 (exact keyword matches) + vector search
(semantic matches), fused with EnsembleRetriever (Reciprocal Rank Fusion).

Same persistence pattern as before: first run embeds and saves the vector
store; every run after that reconnects to the already-saved index instead
of re-embedding everything from scratch.
"""

from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings

from src.config import (
    BM25_WEIGHT,
    CHROMA_PERSIST_DIR,
    DOCS_DIR,
    EMBEDDING_MODEL,
    RETRIEVAL_K,
    VECTOR_WEIGHT,
)
from src.ingestion.loader import load_and_chunk_docs


def _get_vectorstore() -> Chroma:
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    if CHROMA_PERSIST_DIR.exists():
        print(f"Vector store already exists at {CHROMA_PERSIST_DIR} — reconnecting.")
        return Chroma(
            persist_directory=str(CHROMA_PERSIST_DIR),
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
        )

    print("No existing vector store found — chunking docs and embedding for the first time.")
    chunks = load_and_chunk_docs(str(DOCS_DIR))
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(CHROMA_PERSIST_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def get_ensemble_retriever() -> EnsembleRetriever:
    vectorstore = _get_vectorstore()
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    # BM25 needs the raw chunks directly (it isn't backed by the vectorstore at all) —
    # re-chunking here is cheap since it's pure text splitting, no API calls involved.
    chunks = load_and_chunk_docs(str(DOCS_DIR))
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = RETRIEVAL_K

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[BM25_WEIGHT, VECTOR_WEIGHT],
    )


if __name__ == "__main__":
    retriever = get_ensemble_retriever()
    results = retriever.invoke("How do I use dependency injection with yield?")
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ({doc.metadata.get('source')}) ---")
        print(doc.page_content[:200])
