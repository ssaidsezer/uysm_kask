from __future__ import annotations

import os
import time
import uuid
from typing import List, Optional, Sequence

import pdfplumber
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


# ---------------------------------------------------------------------------
# Config  – .env dosyasından veya ortam değişkenlerinden okunur
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.0.149:6333")
DEFAULT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "uysm")

# ---------------------------------------------------------------------------
# Qdrant bağlantısı
# ---------------------------------------------------------------------------
_qdrant_client: Optional[QdrantClient] = None


def get_qdrant_client(url: str = QDRANT_URL) -> QdrantClient:
    """Lazy-init Qdrant bağlantısı."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=url)
    return _qdrant_client


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance: qmodels.Distance = qmodels.Distance.COSINE,
) -> None:
    """Qdrant koleksiyonunu yoksa oluştur, varsa bırak."""
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=vector_size,
            distance=distance,
        ),
    )


# ---------------------------------------------------------------------------
# Ollama Embedding  – uzak Ollama /api/embed endpoint'ini kullanır
# ---------------------------------------------------------------------------
def get_embeddings(
    texts: List[str],
    model: str = OLLAMA_EMBED_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = 120,
) -> List[List[float]]:
    """
    Uzak Ollama sunucusundan embedding al.
    Önce yeni /api/embed, 404 alırsa eski /api/embeddings endpoint'ini dener.
    """
    if not base_url:
        raise ValueError(
            "OLLAMA_BASE_URL ortam değişkeni tanımlı değil. "
            "Lütfen .env dosyasına uzak sunucu adresini ekleyin "
            "(örn: OLLAMA_BASE_URL=http://192.168.1.151:11434)."
        )

    base = base_url.rstrip("/")

    # --- Yeni endpoint: /api/embed (toplu, Ollama >= 0.4) ---
    try:
        resp = requests.post(
            f"{base}/api/embed",
            json={"model": model, "input": texts},
            timeout=timeout,
        )
        if resp.status_code != 404:
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings")
            if embeddings:
                return embeddings
    except requests.exceptions.HTTPError:
        pass  # 404 dışı hata → aşağıda eski endpoint'i dene

    # --- Eski endpoint: /api/embeddings (tek tek, eski Ollama) ---
    all_embeddings: List[List[float]] = []
    for text in texts:
        resp = requests.post(
            f"{base}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        embedding = data.get("embedding")
        if not embedding:
            raise RuntimeError(
                f"Ollama embedding yanıtında 'embedding' alanı bulunamadı. "
                f"Model: {model}, Yanıt: {data}"
            )
        all_embeddings.append(embedding)

    return all_embeddings


# ---------------------------------------------------------------------------
# Chunking yardımcıları
# ---------------------------------------------------------------------------
def _chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[str]:
    """Basit karakter tabanlı chunking."""
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
) -> List[dict]:
    """PDF'den text chunk'larını çıkar."""
    chunks: List[dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""

            for chunk_index, chunk_text_str in enumerate(
                _chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            ):
                chunks.append(
                    {
                        "id": str(uuid.uuid4()),
                        "text": chunk_text_str,
                        "metadata": {
                            "source": pdf_path,
                            "page": page_index,
                            "chunk": chunk_index,
                        },
                    }
                )

    return chunks


# ---------------------------------------------------------------------------
# İndeksleme  – PDF → Ollama embed → Qdrant upsert (zamanlama + ilerleme)
# ---------------------------------------------------------------------------
EMBED_BATCH_SIZE = 50  # Ollama'ya tek seferde gönderilecek metin sayısı
QDRANT_BATCH_SIZE = 100  # Qdrant'a tek seferde gönderilecek point sayısı


def index_pdfs(
    pdf_paths: Sequence[str],
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    qdrant_url: str = QDRANT_URL,
    embed_model: str = OLLAMA_EMBED_MODEL,
    progress_callback=None,
) -> dict:
    """
    PDF'leri Ollama embedding ile embed edip Qdrant'a indeksler.

    progress_callback(phase, current, total, elapsed_sec):
        phase: "pdf_extract" | "ollama_embed" | "qdrant_upsert"
        current, total: ilerleme sayaçları
        elapsed_sec: o adımın süresi

    Returns dict:
        total_chunks, pdf_extract_sec, ollama_embed_sec, qdrant_upsert_sec, total_sec
    """
    timings: dict = {
        "total_chunks": 0,
        "pdf_extract_sec": 0.0,
        "ollama_embed_sec": 0.0,
        "qdrant_upsert_sec": 0.0,
        "total_sec": 0.0,
    }
    total_t0 = time.time()

    # --- 1. PDF Extraction ---
    t0 = time.time()
    all_chunks: List[dict] = []
    for i, path in enumerate(pdf_paths):
        all_chunks.extend(
            _extract_pdf_chunks(
                path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
        elapsed = time.time() - t0
        if progress_callback:
            progress_callback("pdf_extract", i + 1, len(pdf_paths), elapsed)

    timings["pdf_extract_sec"] = round(time.time() - t0, 2)

    if not all_chunks:
        timings["total_sec"] = round(time.time() - total_t0, 2)
        return timings

    texts = [c["text"] for c in all_chunks]
    total_texts = len(texts)
    timings["total_chunks"] = total_texts

    # --- 2. Ollama Embedding (batch'ler halinde) ---
    t0 = time.time()
    all_embeddings: List[List[float]] = []
    for start in range(0, total_texts, EMBED_BATCH_SIZE):
        batch_texts = texts[start : start + EMBED_BATCH_SIZE]
        batch_embeddings = get_embeddings(batch_texts, model=embed_model)
        all_embeddings.extend(batch_embeddings)
        elapsed = time.time() - t0
        done = min(start + EMBED_BATCH_SIZE, total_texts)
        if progress_callback:
            progress_callback("ollama_embed", done, total_texts, elapsed)

    timings["ollama_embed_sec"] = round(time.time() - t0, 2)

    vector_size = len(all_embeddings[0])

    # --- 3. Qdrant Upsert (batch'ler halinde) ---
    client = get_qdrant_client(url=qdrant_url)
    ensure_collection(client, collection_name, vector_size=vector_size)

    points = [
        qmodels.PointStruct(
            id=all_chunks[i]["id"],
            vector=all_embeddings[i],
            payload={
                **all_chunks[i]["metadata"],
                "text": all_chunks[i]["text"],
            },
        )
        for i in range(len(all_chunks))
    ]

    t0 = time.time()
    for start in range(0, len(points), QDRANT_BATCH_SIZE):
        batch = points[start : start + QDRANT_BATCH_SIZE]
        client.upsert(collection_name=collection_name, points=batch)
        elapsed = time.time() - t0
        done = min(start + QDRANT_BATCH_SIZE, len(points))
        if progress_callback:
            progress_callback("qdrant_upsert", done, len(points), elapsed)

    timings["qdrant_upsert_sec"] = round(time.time() - t0, 2)
    timings["total_sec"] = round(time.time() - total_t0, 2)

    return timings


# ---------------------------------------------------------------------------
# Retrieval  – soru → Ollama embed → Qdrant search
# ---------------------------------------------------------------------------
def retrieve_context(
    question: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    k: int = 5,
    qdrant_url: str = QDRANT_URL,
) -> str:
    """
    Soruyu Ollama ile embed edip Qdrant'tan en yakın k chunk'ı döndürür.
    """
    question_embedding = get_embeddings([question])[0]

    client = get_qdrant_client(url=qdrant_url)

    query_result = client.query_points(
        collection_name=collection_name,
        query=question_embedding,
        limit=k,
    )
    results = getattr(query_result, "points", query_result)

    docs: List[str] = []
    for hit in results:
        if isinstance(hit, dict):
            payload = hit.get("payload") or {}
        else:
            payload = getattr(hit, "payload", None) or {}
        text = payload.get("text", "")
        if isinstance(text, str) and text.strip():
            docs.append(text)

    if not docs:
        return ""

    return "\n\n".join(docs)


def retrieve_chunks(
    question: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    k: int = 5,
    qdrant_url: str = QDRANT_URL,
) -> List[str]:
    """
    Soruyu Ollama ile embed edip Qdrant'tan en yakın k chunk'ı liste olarak döndürür.
    """
    question_embedding = get_embeddings([question])[0]

    client = get_qdrant_client(url=qdrant_url)

    query_result = client.query_points(
        collection_name=collection_name,
        query=question_embedding,
        limit=k,
    )
    results = getattr(query_result, "points", query_result)

    docs: List[str] = []
    for hit in results:
        if isinstance(hit, dict):
            payload = hit.get("payload") or {}
        else:
            payload = getattr(hit, "payload", None) or {}
        text = payload.get("text", "")
        if isinstance(text, str) and text.strip():
            docs.append(text)

    return docs
