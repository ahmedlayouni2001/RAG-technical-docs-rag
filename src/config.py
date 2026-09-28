import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "data" / "docs"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "data" / "chroma_db"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"
GRADER_MODEL = "gpt-4o-mini"  # cheaper/faster model for the yes/no relevance grading step

BM25_WEIGHT = 0.5
VECTOR_WEIGHT = 0.5
RETRIEVAL_K = 5
MAX_AGENT_RETRIES = 2

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
