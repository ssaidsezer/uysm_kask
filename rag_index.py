from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


EMBEDDING_MODEL_NAME = "trmteb/turkish-embedding-model"
DEFAULT_CHROMA_DIR = "./chroma_db"
DEFAULT_COLLECTION_NAME = "uysm"


_embedding_model: Optional[SentenceTransformer] = None


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """
    Lazy-load and cache the embedding model.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def get_chroma_client(persist_directory: str = DEFAULT_CHROMA_DIR):
    """
    Create a Chroma client using the modern PersistentClient API when available.

    Falls back to the legacy Settings-based client only if needed.
    """
    # New-style persistent client (Chroma >= 0.5)
    if hasattr(chromadb, "PersistentClient"):
        return chromadb.PersistentClient(path=persist_directory)  # type: ignore[attr-defined]

    # Fallback for very old Chroma versions
    from chromadb.config import Settings

    return chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory,
        )
    )


def get_or_create_collection(
    client: chromadb.Client, name: str = DEFAULT_COLLECTION_NAME
) -> chromadb.api.models.Collection.Collection:
    """
    Get or create a Chroma collection.
    """
    return client.get_or_create_collection(name=name)


@dataclass
class PdfChunk:
    id: str
    text: str
    metadata: dict


def _chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    Simple character-based chunking with overlap.
    """
    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = max(end - chunk_overlap, 0)

    return chunks


def _extract_pdf_chunks(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[PdfChunk]:
    """
    Read a PDF and return text chunks with basic metadata.
    """
    reader = PdfReader(pdf_path)
    chunks: List[PdfChunk] = []

    for page_index, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""

        for chunk_index, chunk_text in enumerate(
            _chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            chunk_id = f"{pdf_path}::page{page_index}::chunk{chunk_index}"
            metadata = {
                "source": pdf_path,
                "page": page_index,
                "chunk": chunk_index,
            }
            chunks.append(PdfChunk(id=chunk_id, text=chunk_text, metadata=metadata))

    return chunks


def index_pdfs(
    pdf_paths: Sequence[str],
    collection: chromadb.api.models.Collection.Collection,
    embedding_model: Optional[SentenceTransformer] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> int:
    """
    Index the given PDFs into the provided Chroma collection.

    Returns the number of chunks indexed.
    """
    if embedding_model is None:
        embedding_model = load_embedding_model()

    all_chunks: List[PdfChunk] = []
    for path in pdf_paths:
        all_chunks.extend(
            _extract_pdf_chunks(
                path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    if not all_chunks:
        return 0

    texts = [c.text for c in all_chunks]
    embeddings = embedding_model.encode(texts, show_progress_bar=False).tolist()

    collection.upsert(
        ids=[c.id for c in all_chunks],
        documents=texts,
        metadatas=[c.metadata for c in all_chunks],
        embeddings=embeddings,
    )

    return len(all_chunks)


def retrieve_context(
    question: str,
    collection: chromadb.api.models.Collection.Collection,
    embedding_model: Optional[SentenceTransformer] = None,
    k: int = 5,
) -> str:
    """
    Retrieve top-k relevant chunks from Chroma and return as a single context string.
    """
    if embedding_model is None:
        embedding_model = load_embedding_model()

    question_embedding = embedding_model.encode([question], show_progress_bar=False)[
        0
    ].tolist()

    result = collection.query(
        query_embeddings=[question_embedding],
        n_results=k,
    )

    docs: List[str] = []
    if result and "documents" in result and result["documents"]:
        docs = result["documents"][0] or []

    docs = [d for d in docs if isinstance(d, str) and d.strip()]
    if not docs:
        return ""

    return "\n\n".join(docs)

