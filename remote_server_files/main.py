from __future__ import annotations

import io
import os
import uuid
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.0.149:6333")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "trmteb/turkish-embedding-model"
)
DEFAULT_COLLECTION_NAME = os.getenv("COLLECTION_NAME", "uysm")

# Chunk ayarları (isteğe göre env ile override edebilirsin)
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
app = FastAPI(title="Remote RAG Index API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # gerekirse daralt
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qdrant_client = QdrantClient(url=QDRANT_URL)
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


# -----------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# -----------------------------------------------------------------------------
def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
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


def ensure_collection(
    collection_name: str,
    vector_size: int,
    distance: qmodels.Distance = qmodels.Distance.COSINE,
) -> None:
    """Qdrant koleksiyonunu yoksa oluştur, varsa bırak."""
    existing = [c.name for c in qdrant_client.get_collections().collections]
    if collection_name in existing:
        return
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=vector_size,
            distance=distance,
        ),
    )


# -----------------------------------------------------------------------------
# API modelleri
# -----------------------------------------------------------------------------
class RetrieveRequest(BaseModel):
    question: str
    collection_name: str = DEFAULT_COLLECTION_NAME
    k: int = 5


class RetrieveResponse(BaseModel):
    context: str
    documents: List[str]


# -----------------------------------------------------------------------------
# Endpoint'ler
# -----------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/index")
async def index_pdfs(
    files: List[UploadFile] = File(..., description="PDF dosyaları"),
    collection_name: str = Form(DEFAULT_COLLECTION_NAME),
    chunk_size: int = Form(DEFAULT_CHUNK_SIZE),
    chunk_overlap: int = Form(DEFAULT_CHUNK_OVERLAP),
) -> dict:
    """
    Gönderilen PDF'leri okuyup Qdrant koleksiyonuna indeksler.
    - PDF'ler multipart/form-data ile upload edilir.
    """
    model = get_embedding_model()

    texts: List[str] = []
    metadatas: List[dict] = []
    ids: List[str] = []

    for f in files:
        raw = await f.read()
        reader = PdfReader(io.BytesIO(raw))

        for page_index, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""

            for chunk_index, chunk in enumerate(
                chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            ):
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)
                texts.append(chunk)
                metadatas.append(
                    {
                        "source": f.filename,
                        "page": page_index,
                        "chunk": chunk_index,
                    }
                )

    if not texts:
        return {"indexed_chunks": 0}

    # Embedding hesapla
    embeddings = model.encode(texts, show_progress_bar=False)
    vector_size = len(embeddings[0])

    # Koleksiyonu hazırla
    ensure_collection(collection_name, vector_size=vector_size)

    points = [
        qmodels.PointStruct(id=ids[i], vector=embeddings[i].tolist(), payload=metadatas[i])
        for i in range(len(ids))
    ]

    qdrant_client.upsert(
        collection_name=collection_name,
        points=points,
    )

    return {"indexed_chunks": len(points), "collection_name": collection_name}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_context(req: RetrieveRequest) -> RetrieveResponse:
    """
    Soruyu embed edip Qdrant'tan top-k dokümanları döndürür.
    """
    model = get_embedding_model()
    question_vec = model.encode([req.question], show_progress_bar=False)[0].tolist()

    res = qdrant_client.search(
        collection_name=req.collection_name,
        query_vector=question_vec,
        limit=req.k,
    )

    docs: List[str] = []
    for hit in res:
        payload = hit.payload or {}
        text = payload.get("text") or payload.get("document")
        # Eğer text yoksa, embedding için girdiğimiz 'document' alanını payload'a da koymak istersen
        if not text and "chunk" in payload:
            # bizim yukarıda sadece metadata'yı payload'a koyduğumuzu varsayarsak
            # metni alamayız; bu durumda docs boş kalır.
            text = ""
        if isinstance(text, str) and text.strip():
            docs.append(text)

    # Eğer metni payload'a koymadıysan, yukarıdaki mantığı
    # embedding eklerken payload'a "text": chunk şeklinde yazarak düzelt.
    context = "\n\n".join(docs) if docs else ""

    return RetrieveResponse(context=context, documents=docs)