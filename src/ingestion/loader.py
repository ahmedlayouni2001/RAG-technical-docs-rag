"""
Phase 1: Ingestion.

Loads markdown documentation, splits it header-by-header (so a chunk never
straddles two unrelated sections), then applies size-based splitting within
each section. Every resulting chunk gets a contextual header prepended —
the file path plus its heading breadcrumb — so a chunk still makes sense
once it's floating alone in the vector store, disconnected from its
original position in the docs.
"""

import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

# Strips MkDocs-style permalink anchors, e.g. "Advanced Dependencies { #advanced-dependencies }"
# -> "Advanced Dependencies". Real documentation is full of small surprises like this.
_ANCHOR_TAG_RE = re.compile(r"\s*\{\s*#[\w-]+\s*\}\s*$")


def _clean_heading(heading: str) -> str:
    return _ANCHOR_TAG_RE.sub("", heading).strip()


def _build_header_breadcrumb(metadata: dict) -> str:
    """Turns {'h1': 'Tutorial', 'h2': 'Path Parameters'} into 'Tutorial > Path Parameters'."""
    parts = [_clean_heading(metadata[key]) for key in ("h1", "h2", "h3") if metadata.get(key)]
    return " > ".join(parts)


def load_and_chunk_docs(
    docs_dir: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> List[Document]:
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # keep the heading text inside the chunk itself, not just in metadata
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    docs_path = Path(docs_dir)
    all_chunks: List[Document] = []

    for md_file in sorted(docs_path.rglob("*.md")):
        raw_text = md_file.read_text(encoding="utf-8")
        if not raw_text.strip():
            continue

        relative_path = str(md_file.relative_to(docs_path))
        
        section_docs = md_splitter.split_text(raw_text)
        
        for section in section_docs:
            section.metadata["source"] = relative_path
        
            
        section_chunks = size_splitter.split_documents(section_docs) 
        
        for chunk in section_chunks:
            
            breadcrumb = _build_header_breadcrumb(chunk.metadata)
            header_line = f"[{relative_path}" + (f" > {breadcrumb}]" if breadcrumb else "]")
            chunk.page_content = f"{header_line}\n\n{chunk.page_content}"

        all_chunks.extend(section_chunks)

    return all_chunks


if __name__ == "__main__":
    chunks = load_and_chunk_docs("data/docs", chunk_size=1000, chunk_overlap=100)

    print(f"Total chunks created: {len(chunks)}")
    print(f"Average chunk length: {sum(len(c.page_content) for c in chunks) // len(chunks)} characters")

    print("\n--- Example chunk ---")
    example = chunks[50]
    print(example.page_content[:500])
    print("\nMetadata:", example.metadata)