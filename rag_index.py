from __future__ import annotations

import json as _json
import os
import re as _re
import time
import hashlib
import uuid
from typing import List, Optional, Sequence

import pdfplumber
import requests
import qdrant_client as _qdrant_pkg
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config  – .env dosyasından veya ortam değişkenlerinden okunur
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.0.149:6333")
DEFAULT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "uysm")
DEBUG_LOG_PATH = os.path.join(os.path.dirname(__file__), "debug-3c80e4.log")
DEBUG_SESSION_ID = "3c80e4"


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        payload = {
            "sessionId": DEBUG_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as _f:
            _f.write(_json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass

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


def _normalize_text_for_dedup(text: str) -> str:
    """Whitespace farklarını normalize ederek metin karşılaştır."""
    return " ".join(text.split()).strip()


def _chunk_id(source: str, page: int, chunk_index: int) -> str:
    """
    Aynı PDF tekrar indekslendiğinde point ID sabit kalsın.
    Böylece Qdrant upsert duplicate eklemek yerine güncelleme yapar.
    """
    raw = f"{source}|{page}|{chunk_index}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance: qmodels.Distance = qmodels.Distance.COSINE,
) -> None:
    """Qdrant koleksiyonunu yoksa oluştur, dimension değiştiyse sil ve yeniden oluştur."""
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        info = client.get_collection(collection_name)
        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, dict):
            existing_size = next(iter(vectors_config.values())).size
        else:
            existing_size = vectors_config.size if vectors_config else None
        if existing_size == vector_size:
            return
        # Dimension değişti — eski collection'ı sil ve yeniden oluştur
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=vector_size,
            distance=distance,
        ),
    )


def _qdrant_query_points_compat(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    limit: int,
    score_threshold: float,
    run_id: str,
    location: str,
) -> list:
    #region agent log
    _debug_log(
        run_id,
        "H2",
        location,
        "qdrant_query_dispatch",
        {
            "has_query_points": hasattr(client, "query_points"),
            "has_search": hasattr(client, "search"),
            "collection_name": collection_name,
            "limit": limit,
            "score_threshold": score_threshold,
            "query_dim": len(query_vector),
        },
    )
    #endregion
    if hasattr(client, "query_points"):
        query_result = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )
        return getattr(query_result, "points", query_result)

    if hasattr(client, "search"):
        return client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )

    raise AttributeError("Qdrant client has neither query_points nor search methods.")


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

def _snap_to_word_boundary(text: str, pos: int) -> int:
    """
    pos karakterinden geriye doğru en yakın boşluk/satır sonunu bulur ve
    kesim noktasını o boşluğun hemen sonrasına (bir sonraki kelime başına)
    taşır.  Böylece kelimeler hiçbir zaman ortasından kesilmez.

    Maksimum 200 karakter geri gider; uygun nokta bulunamazsa pos döner.
    """
    if pos <= 0 or pos >= len(text):
        return pos
    # Zaten boşluk veya satır başındaysa düzeltme gerekmez
    if text[pos] in (" ", "\n", "\t"):
        return pos
    for i in range(pos, max(pos - 200, 0), -1):
        if text[i] in (" ", "\n", "\t"):
            return i + 1
    return pos


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
        if end < length:
            end = _snap_to_word_boundary(text, end)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - chunk_overlap, 0)

    return chunks


def _decode_pdf_text(raw: str) -> str:
    """PDF'den gelen metindeki encoding bozukluklarını temizler.

    pdfplumber bazen UTF-8 byte'larını latin-1 olarak yorumlar (mojibake).
    Örn: 'Ä°' → 'İ'. Sadece bu durum tespit edildiğinde dönüştürme yapılır;
    metin zaten geçerli unicode ise olduğu gibi döndürülür.
    """
    if not raw:
        return ""
    # Mojibake tespiti: latin-1 encode edilebiliyor ve UTF-8 decode başarılı ise
    # orijinal metin bozulmuş demektir, düzelt.
    try:
        fixed = raw.encode("latin-1").decode("utf-8")
        # Ek kontrol: düzeltilmiş metinde Türkçe karakterler varsa mojibake'di
        if any(c in fixed for c in "çğıöşüÇĞİÖŞÜ"):
            return fixed
        # Orijinalde de Türkçe karakter varsa zaten doğruydu
        if any(c in raw for c in "çğıöşüÇĞİÖŞÜ"):
            return raw
        return fixed
    except (UnicodeEncodeError, UnicodeDecodeError):
        # latin-1 encode başarısız → metin zaten unicode, dokunma
        return raw


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
                page_text = _decode_pdf_text(page_text)
            except Exception:
                page_text = ""

            for chunk_index, chunk_text_str in enumerate(
                _chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            ):
                chunks.append(
                    {
                        "id": _chunk_id(pdf_path, page_index, chunk_index),
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

    # Aynı source daha önce indekslendiyse eski point'leri temizle.
    # Böylece tekrar indekslemede duplicate birikmez.
    unique_sources = sorted({str(path) for path in pdf_paths if str(path).strip()})
    if unique_sources:
        source_filter = qmodels.Filter(
            should=[
                qmodels.FieldCondition(
                    key="source",
                    match=qmodels.MatchValue(value=source),
                )
                for source in unique_sources
            ]
        )
        client.delete(
            collection_name=collection_name,
            points_selector=qmodels.FilterSelector(filter=source_filter),
            wait=True,
        )

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
    score_threshold: float = 0.55,
) -> str:
    """
    Soruyu Ollama ile embed edip Qdrant'tan en yakın k chunk'ı döndürür.
    """
    run_id = f"retrieve_context_{int(time.time() * 1000)}"
    question_embedding = get_embeddings([question])[0]

    client = get_qdrant_client(url=qdrant_url)
    #region agent log
    _debug_log(
        run_id,
        "H1",
        "rag_index.py:retrieve_context:before_query",
        "qdrant_client_capabilities",
        {
            "qdrant_version": getattr(_qdrant_pkg, "__version__", "unknown"),
            "client_type": type(client).__name__,
            "has_query_points": hasattr(client, "query_points"),
            "has_search": hasattr(client, "search"),
            "embedding_dim": len(question_embedding) if isinstance(question_embedding, list) else -1,
            "collection_name": collection_name,
        },
    )
    #endregion

    results = _qdrant_query_points_compat(
        client=client,
        collection_name=collection_name,
        query_vector=question_embedding,
        limit=k,
        score_threshold=score_threshold,
        run_id=run_id,
        location="rag_index.py:retrieve_context:query_dispatch",
    )

    docs: List[str] = []
    seen: set[str] = set()
    for hit in results:
        if isinstance(hit, dict):
            payload = hit.get("payload") or {}
        else:
            payload = getattr(hit, "payload", None) or {}
        text = payload.get("text", "")
        if isinstance(text, str) and text.strip():
            key = _normalize_text_for_dedup(text)
            if key and key not in seen:
                seen.add(key)
                docs.append(text)

    if not docs:
        return ""

    return "\n\n".join(docs)


def retrieve_chunks(
    question: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    k: int = 5,
    qdrant_url: str = QDRANT_URL,
    score_threshold: float = 0.55,
    embed_model: str | None = None,
) -> List[dict]:
    """
    Soruyu Ollama ile embed edip Qdrant'tan en yakın k chunk'ı döndürür.
    Her eleman {"text": str, "score": float} dict'idir.
    """
    run_id = f"retrieve_chunks_{int(time.time() * 1000)}"
    question_embedding = get_embeddings([question], model=embed_model or OLLAMA_EMBED_MODEL)[0]

    client = get_qdrant_client(url=qdrant_url)
    #region agent log
    _debug_log(
        run_id,
        "H1",
        "rag_index.py:retrieve_chunks:before_query",
        "qdrant_client_capabilities",
        {
            "qdrant_version": getattr(_qdrant_pkg, "__version__", "unknown"),
            "client_type": type(client).__name__,
            "has_query_points": hasattr(client, "query_points"),
            "has_search": hasattr(client, "search"),
            "embedding_dim": len(question_embedding) if isinstance(question_embedding, list) else -1,
            "collection_name": collection_name,
            "embed_model": embed_model or OLLAMA_EMBED_MODEL,
        },
    )
    #endregion

    results = _qdrant_query_points_compat(
        client=client,
        collection_name=collection_name,
        query_vector=question_embedding,
        limit=k,
        score_threshold=score_threshold,
        run_id=run_id,
        location="rag_index.py:retrieve_chunks:query_dispatch",
    )

    docs: List[dict] = []
    seen: set[str] = set()
    for hit in results:
        if isinstance(hit, dict):
            payload = hit.get("payload") or {}
            score = hit.get("score", 0.0)
        else:
            payload = getattr(hit, "payload", None) or {}
            score = getattr(hit, "score", 0.0)
        text = payload.get("text", "")
        if isinstance(text, str) and text.strip():
            key = _normalize_text_for_dedup(text)
            if key and key not in seen:
                seen.add(key)
                docs.append({"text": text, "score": score})

    return docs


# ===========================================================================
# SMART CHUNKING  – Hiyerarşik + LLM hibrit RAG
# ===========================================================================

SMART_PARENT_BLOCK_SIZE: int = int(os.getenv("SMART_PARENT_SIZE", "3000"))
SMART_CHILD_SIZE: int = int(os.getenv("SMART_CHILD_SIZE", "500"))
SMART_CHILD_OVERLAP: int = int(os.getenv("SMART_CHILD_OVERLAP", "100"))
# REMOVED: fixed 2-block window replaces greedy window
# SMART_MAX_WINDOW_BLOCKS: int = int(os.getenv("SMART_MAX_WINDOW", "4"))
SMART_BOUNDARY_LLM_MODEL: str = os.getenv("BOUNDARY_LLM_MODEL", "qwen3:1.7b")


# ---------------------------------------------------------------------------
# ID üreticiler
# ---------------------------------------------------------------------------

def _parent_block_id(source: str, block_idx: int) -> str:
    raw = f"parent|{source}|{block_idx}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


def _child_block_id(parent_id: str, child_idx: int) -> str:
    raw = f"child|{parent_id}|{child_idx}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


# ---------------------------------------------------------------------------
# PDF tam metin çıkarma + temizleme
# ---------------------------------------------------------------------------

_RE_SEPARATOR  = _re.compile(r'^[\-=_\.~\*#]{3,}\s*$', _re.MULTILINE)
_RE_PAGE_NUM   = _re.compile(r'^\s*(?:Sayfa\s*)?\d+\s*$|^\s*-\s*\d+\s*-\s*$', _re.MULTILINE | _re.IGNORECASE)
_RE_SHORT_LINE = _re.compile(r'^\s{0,2}[^\w\s]{0,2}\s*$', _re.MULTILINE)
_RE_CTRL_CHAR  = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_RE_MULTI_NL   = _re.compile(r'\n{3,}')


def _clean_pdf_text(text: str) -> str:
    """pdfplumber çıktısındaki gürültüyü temizler."""
    text = _RE_CTRL_CHAR.sub('', text)
    text = _RE_SEPARATOR.sub('', text)
    text = _RE_PAGE_NUM.sub('', text)
    text = _RE_SHORT_LINE.sub('', text)
    text = _RE_MULTI_NL.sub('\n\n', text)
    return text.strip()


def _extract_full_pdf_text(pdf_path: str) -> str:
    """PDF'teki tüm sayfaların metnini tek bir string olarak döndürür."""
    parts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            try:
                text = page.extract_text() or ""
                text = _decode_pdf_text(text)
            except Exception:
                text = ""
            if text.strip():
                parts.append(text)
    return _clean_pdf_text("\n".join(parts))


# ---------------------------------------------------------------------------
# İlk sabit-boyut parent blok bölme
# ---------------------------------------------------------------------------

def _split_into_parent_blocks(
    text: str,
    parent_size: int = SMART_PARENT_BLOCK_SIZE,
) -> List[str]:
    """Tüm metni örtüşmesiz sabit boyutlu büyük bloklara böler."""
    text = text.strip()
    if not text:
        return []
    blocks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + parent_size, len(text))
        if end < len(text):
            end = _snap_to_word_boundary(text, end)
        block = text[start:end].strip()
        if block:
            blocks.append(block)
        if end >= len(text):
            break
        start = end
    return blocks


# ---------------------------------------------------------------------------
# Qdrant parent koleksiyonu (dummy 1D vektör – sadece payload deposu)
# ---------------------------------------------------------------------------

def _ensure_parent_collection(client: QdrantClient, collection_name: str) -> None:
    """
    Parent blokları için Qdrant koleksiyonu oluşturur.
    Boyutu 1 olan dummy vektör kullanılır; parent'lar ID ile doğrudan çekilir.
    """
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=1,
                distance=qmodels.Distance.COSINE,
            ),
        )


# ---------------------------------------------------------------------------
# LLM ile semantik sınır tespiti
# ---------------------------------------------------------------------------

def _line_index_to_char_offset(
    block_a: str,
    block_b: str,
    line_index: int,
) -> Optional[int]:
    """
    Global line_index'i (block_a + '\n' + block_b içinde) karakter offset'ine çevirir.
    line_index 0 geçersiz bölünme noktasıdır.
    Döndürür: char offset veya None (güvenli sınırlar dışında ise).
    """
    combined = block_a + "\n" + block_b
    lines = combined.split("\n")
    if line_index < 1 or line_index >= len(lines):
        return None
    offset = sum(len(lines[i]) + 1 for i in range(line_index))
    if offset < 50 or offset > len(combined) - 50:
        return None
    return offset


def _call_llm_for_boundary(
    block_a: str,
    block_b: str,
    base_url: str,
    model: str,
    timeout: int = 120,
) -> tuple[Optional[int], bool]:
    """
    block_a ve block_b'yi satır bazlı JSON olarak LLM'e gönderir.
    Ollama /api/generate + format schema ile yapılandırılmış JSON yanıt alır.
    Döndürür: (global_line_index veya None, timed_out)
    """
    combined = block_a + "\n" + block_b
    raw_lines = combined.split("\n")
    total_lines = len(raw_lines)
    last_line_index = total_lines - 1

    line_entries = [{"index": i, "text": raw_lines[i]} for i in range(total_lines)]
    input_payload = {"lines": line_entries}

    system_prompt = (
        "You are a document segmentation assistant.\n"
        "You will receive a JSON object with a 'lines' array (index + text pairs).\n"
        "These lines come from two consecutive PDF blocks.\n\n"
        "If there is a clear NEW TOPIC or NEW SECTION: set indeks to the first line "
        "of the new topic, status=\"boundary_found\".\n"
        "If single topic throughout: indeks=-1, status=\"no_boundary\".\n"
        "If unsure: indeks=-1, status=\"uncertain\".\n\n"
        f"indeks must be between 1 and {last_line_index}.\n"
        "Prefer paragraph breaks or section headings as boundaries."
    )

    output_schema = {
        "type": "object",
        "properties": {
            "indeks": {"type": "integer"},
            "status": {"type": "string"},
        },
        "required": ["indeks", "status"],
        "additionalProperties": False,
    }

    call_id = f"boundary_{int(time.time() * 1000)}"
    full_prompt = f"{system_prompt}\n\n{_json.dumps(input_payload, ensure_ascii=False)}"

    _debug_log(
        call_id, "boundary_call",
        "rag_index.py:_call_llm_for_boundary:input",
        "llm_input",
        {
            "model": model,
            "total_lines": total_lines,
            "block_a_preview": block_a[:120].replace("\n", "\\n"),
            "block_b_preview": block_b[:120].replace("\n", "\\n"),
            "prompt": full_prompt,
            "input_payload": input_payload,
        },
    )

    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": 0.0},
                "format": output_schema,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.Timeout, requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectTimeout):
        _debug_log(call_id, "boundary_call", "rag_index.py:_call_llm_for_boundary:timeout", "llm_timeout", {})
        return None, True
    except Exception as e:
        _debug_log(call_id, "boundary_call", "rag_index.py:_call_llm_for_boundary:error", "llm_error", {"error": str(e)})
        return None, False

    content = ""
    if isinstance(data, dict):
        content = str(data.get("response") or "")

    _debug_log(
        call_id, "boundary_call",
        "rag_index.py:_call_llm_for_boundary:raw_response",
        "llm_raw_output",
        {"raw": content},
    )

    # <think>...</think> wrapper'larını temizle
    clean = _re.sub(r'<think>.*?</think>', '', content, flags=_re.DOTALL).strip()

    try:
        parsed = _json.loads(clean)
    except (_json.JSONDecodeError, ValueError):
        _debug_log(call_id, "boundary_call", "rag_index.py:_call_llm_for_boundary:parse_fail", "llm_parse_error", {"clean": clean})
        return None, False

    indeks = parsed.get("indeks", -1)
    status = parsed.get("status", "")

    _debug_log(
        call_id, "boundary_call",
        "rag_index.py:_call_llm_for_boundary:result",
        "llm_decision",
        {
            "indeks": indeks,
            "status": status,
            "boundary_found": status == "boundary_found" and isinstance(indeks, int) and 1 <= indeks < total_lines,
        },
    )

    if status in ("no_boundary", "uncertain"):
        return None, False

    if not isinstance(indeks, int) or indeks == -1:
        return None, False

    if indeks < 1 or indeks >= total_lines:
        return None, False

    return indeks, False


def _refine_boundaries_with_llm(
    initial_blocks: List[str],
    base_url: str,
    model: str,
    parent_size: int = SMART_PARENT_BLOCK_SIZE,
    timeout: int = 120,
    progress_callback=None,
) -> tuple[List[str], List[dict]]:
    """
    Strict 2-block pairwise semantik sınır tespiti.

    Her iterasyonda:
      - block_a: pending (önceki iterasyonun ikinci parçası) veya sıradaki blok
      - block_b: sıradaki blok
      - LLM JSON çıktısından global satır indeksi alınır
      - Sınır bulunursa: combined[:offset] → finalize, combined[offset:] → pending
      - Sınır yoksa veya offset geçersizse: block_a + block_b → merge & finalize

    Döndürür: (refined_blocks, timeout_records)
      timeout_records: [{"block_idx": int, "preview": str, "full_text": str}, ...]
    """
    if not initial_blocks:
        return [], []

    refined: List[str] = []
    timeout_records: List[dict] = []
    total_initial = len(initial_blocks)
    hard_cap_chars = max(parent_size * 2 + 200, parent_size + 500)

    def _append_with_fallback_split(text: str) -> None:
        clean = text.strip()
        if not clean:
            return
        if len(clean) <= hard_cap_chars:
            refined.append(clean)
            return
        parts = _split_into_parent_blocks(clean, parent_size=parent_size)
        refined.extend(parts if parts else [clean])

    pending = ""
    block_ptr = 0

    while block_ptr < total_initial or pending:
        # block_a: pending varsa onu kullan, yoksa sıradaki bloğu al
        if pending:
            block_a = pending
            pending = ""
        else:
            block_a = initial_blocks[block_ptr]
            block_ptr += 1

        # block_b yoksa block_a'yı finalizeleyip bitir
        if block_ptr >= total_initial:
            _append_with_fallback_split(block_a)
            if progress_callback:
                progress_callback("llm_boundary", total_initial, total_initial, 0)
            break

        block_b = initial_blocks[block_ptr]
        block_ptr += 1

        # LLM'i tam 2 blokla çağır
        line_idx, timed_out = _call_llm_for_boundary(
            block_a, block_b, base_url, model, timeout
        )

        if timed_out:
            combined_text = block_a + "\n" + block_b
            timeout_records.append({
                "block_idx": block_ptr - 1,
                "preview": combined_text[:120].replace("\n", " "),
                "full_text": combined_text.strip(),
            })

        if line_idx is not None:
            char_offset = _line_index_to_char_offset(block_a, block_b, line_idx)
            if char_offset is not None:
                combined = block_a + "\n" + block_b
                first = combined[:char_offset]
                second = combined[char_offset:]
                _append_with_fallback_split(first)
                pending = second.strip()
            else:
                # Offset güvenli sınırlar dışında → merge
                _append_with_fallback_split(block_a + "\n" + block_b)
        else:
            # Sınır yok → merge
            _append_with_fallback_split(block_a + "\n" + block_b)

        if progress_callback:
            progress_callback("llm_boundary", min(block_ptr, total_initial), total_initial, 0)

    return refined, timeout_records


# ---------------------------------------------------------------------------
# Child chunk bölme
# ---------------------------------------------------------------------------

def _split_children_from_parent(
    parent_text: str,
    parent_id: str,
    source: str,
    block_idx: int,
    child_size: int = SMART_CHILD_SIZE,
    child_overlap: int = SMART_CHILD_OVERLAP,
) -> List[dict]:
    """Semantik parent bloğu sabit boyutlu child chunk'lara böler."""
    children: List[dict] = []
    child_idx = 0
    start = 0
    text = parent_text.strip()

    while start < len(text):
        end = min(start + child_size, len(text))
        if end < len(text):
            end = _snap_to_word_boundary(text, end)
        chunk = text[start:end].strip()
        if chunk:
            children.append(
                {
                    "id": _child_block_id(parent_id, child_idx),
                    "text": chunk,
                    "parent_id": parent_id,
                    "metadata": {
                        "source": source,
                        "block_idx": block_idx,
                        "child_idx": child_idx,
                    },
                }
            )
            child_idx += 1
        if end >= len(text):
            break
        start = max(end - child_overlap, 0)

    return children


# ---------------------------------------------------------------------------
# Smart İndeksleme
# ---------------------------------------------------------------------------

def index_pdfs_smart(
    pdf_paths: Sequence[str],
    base_collection: str = DEFAULT_COLLECTION_NAME,
    parent_size: int = SMART_PARENT_BLOCK_SIZE,
    child_size: int = SMART_CHILD_SIZE,
    child_overlap: int = SMART_CHILD_OVERLAP,
    boundary_llm_model: str = SMART_BOUNDARY_LLM_MODEL,
    embed_model: str = OLLAMA_EMBED_MODEL,
    qdrant_url: str = QDRANT_URL,
    progress_callback=None,
) -> dict:
    """
    Smart chunking pipeline:
        PDF → büyük parent bloklar → LLM semantik sınır tespiti (greedy window)
        → parent bloklar Qdrant'a kaydedilir
        → child chunk'lar embed edilip ayrı koleksiyona yazılır

    Oluşturulan koleksiyonlar:
        {base_collection}_parents  — semantik parent bloklar (ID ile çekilir)
        {base_collection}_children — child chunk embedding'leri (vektör arama)

    progress_callback(phase, current, total, elapsed_sec):
        phase: "pdf_extract" | "llm_boundary" | "parent_store" | "child_embed" | "child_upsert"
    """
    parents_col = f"{base_collection}_parents"
    children_col = f"{base_collection}_children"

    timings: dict = {
        "total_parents": 0,
        "total_children": 0,
        "pdf_extract_sec": 0.0,
        "llm_boundary_sec": 0.0,
        "parent_store_sec": 0.0,
        "child_embed_sec": 0.0,
        "child_upsert_sec": 0.0,
        "total_sec": 0.0,
        "timeout_count": 0,
        "timeout_blocks": [],
    }
    total_t0 = time.time()
    client = get_qdrant_client(url=qdrant_url)
    run_id = f"smart_{int(time.time() * 1000)}"
    #region agent log
    _debug_log(
        run_id,
        "H1",
        "rag_index.py:index_pdfs_smart:start",
        "smart_index_start",
        {
            "pdf_count": len(pdf_paths),
            "base_collection": base_collection,
            "parent_size": parent_size,
            "child_size": child_size,
            "child_overlap": child_overlap,
            "embed_model": embed_model,
            "boundary_llm_model": boundary_llm_model,
        },
    )
    #endregion

    # ------------------------------------------------------------------ #
    # 1. PDF extraction → ilk sabit boyutlu parent bloklar                #
    # ------------------------------------------------------------------ #
    t0 = time.time()
    source_blocks: dict = {}  # source → List[str]
    for i, path in enumerate(pdf_paths):
        full_text = _extract_full_pdf_text(path)
        source_blocks[path] = _split_into_parent_blocks(full_text, parent_size=parent_size)
        if progress_callback:
            progress_callback("pdf_extract", i + 1, len(pdf_paths), time.time() - t0)
    timings["pdf_extract_sec"] = round(time.time() - t0, 2)
    #region agent log
    _debug_log(
        run_id,
        "H2",
        "rag_index.py:index_pdfs_smart:post_extract",
        "source_blocks_ready",
        {
            "sources": len(source_blocks),
            "total_initial_blocks": sum(len(v) for v in source_blocks.values()),
            "empty_source_count": sum(1 for v in source_blocks.values() if not v),
        },
    )
    #endregion

    if not any(source_blocks.values()):
        timings["total_sec"] = round(time.time() - total_t0, 2)
        return timings

    # ------------------------------------------------------------------ #
    # Koleksiyonları önceden hazırla                                      #
    # ------------------------------------------------------------------ #
    _ensure_parent_collection(client, parents_col)

    # children_col için vector boyutu ilk embedding'den öğrenilecek
    children_col_ready = False

    # ------------------------------------------------------------------ #
    # 2-6. Kaynak bazlı streaming: LLM → parent upsert → child embed/upsert
    # ------------------------------------------------------------------ #
    total_all_blocks = sum(len(v) for v in source_blocks.values())
    _block_offset = [0]

    llm_t0 = time.time()
    parent_store_acc = 0.0
    child_embed_acc = 0.0
    child_upsert_acc = 0.0

    for source, blocks in source_blocks.items():
        if not blocks:
            _block_offset[0] += len(blocks)
            continue
        source_parent_written = 0
        source_children_written = 0
        source_empty_parent_count = 0
        #region agent log
        _debug_log(
            run_id,
            "H3",
            "rag_index.py:index_pdfs_smart:source_begin",
            "source_processing_start",
            {
                "source": os.path.basename(source),
                "block_count": len(blocks),
                "children_col_ready": children_col_ready,
            },
        )
        #endregion

        # Bu kaynağa ait eski kayıtları sil (parent + children)
        src_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(key="source", match=qmodels.MatchValue(value=source))]
        )
        client.delete(
            collection_name=parents_col,
            points_selector=qmodels.FilterSelector(filter=src_filter),
            wait=True,
        )
        if children_col_ready:
            client.delete(
                collection_name=children_col,
                points_selector=qmodels.FilterSelector(filter=src_filter),
                wait=True,
            )

        # LLM sınır tespiti — bu kaynak için
        base_url = OLLAMA_BASE_URL

        def _make_inner_cb(off: int, glob_total: int, t_ref: float):
            def _inner(phase, current, total, _elapsed):
                if progress_callback:
                    progress_callback(
                        "llm_boundary",
                        off + current,
                        glob_total,
                        time.time() - t_ref,
                    )
            return _inner

        if base_url:
            inner_cb = _make_inner_cb(_block_offset[0], total_all_blocks, llm_t0)
            refined, src_timeouts = _refine_boundaries_with_llm(
                initial_blocks=blocks,
                base_url=base_url,
                model=boundary_llm_model,
                parent_size=parent_size,
                timeout=120,
                progress_callback=inner_cb,
            )
            for rec in src_timeouts:
                rec["source"] = source
            timings["timeout_blocks"].extend(src_timeouts)
        else:
            refined = blocks
            if progress_callback:
                progress_callback(
                    "llm_boundary",
                    _block_offset[0] + len(blocks),
                    total_all_blocks,
                    time.time() - llm_t0,
                )
        #region agent log
        _debug_log(
            run_id,
            "H2",
            "rag_index.py:index_pdfs_smart:post_refine",
            "refine_completed",
            {
                "source": os.path.basename(source),
                "input_blocks": len(blocks),
                "refined_blocks": len(refined),
                "timeouts_for_source": len([r for r in timings["timeout_blocks"] if r.get("source") == source]),
            },
        )
        #endregion

        _block_offset[0] += len(blocks)

        # Her refined parent için: upsert → child split → embed → upsert
        for block_idx, rtext in enumerate(refined):
            if not rtext.strip():
                source_empty_parent_count += 1
                continue

            pid = _parent_block_id(source, block_idx)

            # --- Parent Qdrant'a yaz ---
            t_ps = time.time()
            client.upsert(
                collection_name=parents_col,
                points=[
                    qmodels.PointStruct(
                        id=pid,
                        vector=[0.0],  # dummy — parent'lar ID ile çekilir
                        payload={
                            "text": rtext,
                            "source": source,
                            "block_idx": block_idx,
                        },
                    )
                ],
            )
            parent_store_acc += time.time() - t_ps
            timings["total_parents"] += 1
            source_parent_written += 1
            if block_idx < 3:
                #region agent log
                _debug_log(
                    run_id,
                    "H4",
                    "rag_index.py:index_pdfs_smart:parent_upsert",
                    "parent_upserted",
                    {
                        "source": os.path.basename(source),
                        "block_idx": block_idx,
                        "parent_id": pid,
                        "text_len": len(rtext),
                    },
                )
                #endregion

            if progress_callback:
                progress_callback(
                    "parent_store",
                    timings["total_parents"],
                    timings["total_parents"],
                    parent_store_acc,
                )

            # --- Child'ları böl + embed + upsert ---
            children = _split_children_from_parent(
                parent_text=rtext,
                parent_id=pid,
                source=source,
                block_idx=block_idx,
                child_size=child_size,
                child_overlap=child_overlap,
            )
            if not children:
                if block_idx < 3:
                    #region agent log
                    _debug_log(
                        run_id,
                        "H5",
                        "rag_index.py:index_pdfs_smart:children_split",
                        "children_split_empty",
                        {
                            "source": os.path.basename(source),
                            "block_idx": block_idx,
                            "parent_text_len": len(rtext),
                        },
                    )
                    #endregion
                continue

            child_texts = [c["text"] for c in children]
            child_embeddings: List[List[float]] = []

            t_ce = time.time()
            for start in range(0, len(child_texts), EMBED_BATCH_SIZE):
                batch = child_texts[start: start + EMBED_BATCH_SIZE]
                child_embeddings.extend(get_embeddings(batch, model=embed_model))
            child_embed_acc += time.time() - t_ce
            if block_idx < 3:
                #region agent log
                _debug_log(
                    run_id,
                    "H5",
                    "rag_index.py:index_pdfs_smart:child_embed",
                    "child_embeddings_ready",
                    {
                        "source": os.path.basename(source),
                        "block_idx": block_idx,
                        "children_count": len(children),
                        "embedding_count": len(child_embeddings),
                        "first_embedding_dim": len(child_embeddings[0]) if child_embeddings else 0,
                    },
                )
                #endregion

            timings["total_children"] += len(children)

            if progress_callback:
                progress_callback(
                    "child_embed",
                    timings["total_children"],
                    timings["total_children"],
                    child_embed_acc,
                )

            # children_col'u ilk embedding boyutuyla lazy init et
            if not children_col_ready and child_embeddings:
                vector_size = len(child_embeddings[0])
                ensure_collection(client, children_col, vector_size=vector_size)
                children_col_ready = True
                #region agent log
                _debug_log(
                    run_id,
                    "H4",
                    "rag_index.py:index_pdfs_smart:children_collection",
                    "children_collection_initialized",
                    {
                        "source": os.path.basename(source),
                        "vector_size": vector_size,
                        "children_col": children_col,
                    },
                )
                #endregion
                # Bu kaynağın eski children verisini şimdi silebiliriz
                client.delete(
                    collection_name=children_col,
                    points_selector=qmodels.FilterSelector(filter=src_filter),
                    wait=True,
                )

            if not children_col_ready:
                continue

            child_points = [
                qmodels.PointStruct(
                    id=children[i]["id"],
                    vector=child_embeddings[i],
                    payload={
                        **children[i]["metadata"],
                        "text": children[i]["text"],
                        "parent_id": children[i]["parent_id"],
                    },
                )
                for i in range(len(children))
            ]

            t_cu = time.time()
            for start in range(0, len(child_points), QDRANT_BATCH_SIZE):
                client.upsert(
                    collection_name=children_col,
                    points=child_points[start: start + QDRANT_BATCH_SIZE],
                )
            child_upsert_acc += time.time() - t_cu
            source_children_written += len(child_points)

            if progress_callback:
                progress_callback(
                    "child_upsert",
                    timings["total_children"],
                    timings["total_children"],
                    child_upsert_acc,
                )
        #region agent log
        _debug_log(
            run_id,
            "H3",
            "rag_index.py:index_pdfs_smart:source_end",
            "source_processing_end",
            {
                "source": os.path.basename(source),
                "parents_written": source_parent_written,
                "children_written": source_children_written,
                "empty_parent_count": source_empty_parent_count,
                "children_col_ready": children_col_ready,
            },
        )
        #endregion

    timings["llm_boundary_sec"] = round(time.time() - llm_t0, 2)
    timings["parent_store_sec"] = round(parent_store_acc, 2)
    timings["child_embed_sec"] = round(child_embed_acc, 2)
    timings["child_upsert_sec"] = round(child_upsert_acc, 2)
    timings["timeout_count"] = len(timings["timeout_blocks"])
    timings["total_sec"] = round(time.time() - total_t0, 2)
    return timings


# ---------------------------------------------------------------------------
# Smart Retrieval – child eşleşmesi → parent bağlamı
# ---------------------------------------------------------------------------

def retrieve_chunks_smart(
    question: str,
    base_collection: str = DEFAULT_COLLECTION_NAME,
    k: int = 5,
    qdrant_url: str = QDRANT_URL,
    score_threshold: float = 0.55,
    embed_model: str | None = None,
) -> List[dict]:
    """
    Smart retrieval: children koleksiyonunda vektör araması yapar,
    her eşleşen child chunk için parent_id ile tam parent bloğu Qdrant'tan çeker.

    Dönüş: [{"text": parent_text, "child_text": child_text, "score": float, "parent_id": str}]
    Modele verilen bağlam parent_text'tir (büyük, semantik bütün).
    """
    children_col = f"{base_collection}_children"
    parents_col = f"{base_collection}_parents"

    run_id = f"retrieve_chunks_smart_{int(time.time() * 1000)}"
    question_embedding = get_embeddings([question], model=embed_model or OLLAMA_EMBED_MODEL)[0]
    client = get_qdrant_client(url=qdrant_url)

    results = _qdrant_query_points_compat(
        client=client,
        collection_name=children_col,
        query_vector=question_embedding,
        limit=k,
        score_threshold=score_threshold,
        run_id=run_id,
        location="rag_index.py:retrieve_chunks_smart:query_dispatch",
    )

    # Tekil parent_id'leri skora göre sıralı tut
    parent_ids_ordered: List[str] = []
    child_info: dict = {}
    seen_parents: set = set()

    for hit in results:
        if isinstance(hit, dict):
            payload = hit.get("payload") or {}
            score = hit.get("score", 0.0)
        else:
            payload = getattr(hit, "payload", None) or {}
            score = getattr(hit, "score", 0.0)

        child_text = payload.get("text", "")
        parent_id = str(payload.get("parent_id", ""))

        if parent_id and parent_id not in seen_parents:
            seen_parents.add(parent_id)
            parent_ids_ordered.append(parent_id)
            child_info[parent_id] = {"child_text": child_text, "score": score}

    if not parent_ids_ordered:
        return []

    # Parent blokları ID ile doğrudan çek
    try:
        parent_records = client.retrieve(
            collection_name=parents_col,
            ids=parent_ids_ordered,
            with_payload=True,
        )
    except Exception:
        return []

    parent_text_map: dict = {}
    for rec in parent_records:
        if isinstance(rec, dict):
            pid = str(rec.get("id", ""))
            payload = rec.get("payload") or {}
        else:
            pid = str(getattr(rec, "id", ""))
            payload = getattr(rec, "payload", None) or {}
        if pid:
            parent_text_map[pid] = payload.get("text", "")

    docs: List[dict] = []
    seen_text: set = set()

    for parent_id in parent_ids_ordered:
        parent_text = parent_text_map.get(parent_id, "")
        info = child_info[parent_id]
        if parent_text.strip():
            key = _normalize_text_for_dedup(parent_text)
            if key not in seen_text:
                seen_text.add(key)
                docs.append(
                    {
                        "text": parent_text,
                        "child_text": info["child_text"],
                        "score": info["score"],
                        "parent_id": parent_id,
                    }
                )

    return docs


# ---------------------------------------------------------------------------
# BM25 Retrieval
# ---------------------------------------------------------------------------

def _bm25_tokenize(text: str) -> List[str]:
    """Basit lowercase whitespace tokenizer, Türkçe dahil."""
    return text.lower().split()


def _scroll_all_payloads(client: QdrantClient, collection_name: str) -> List[dict]:
    """Bir koleksiyondaki tüm payload'ları sayfalayarak döndürür."""
    payloads: List[dict] = []
    offset = None
    while True:
        result, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=None,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for rec in result:
            if isinstance(rec, dict):
                pid = str(rec.get("id", ""))
                payload = rec.get("payload") or {}
            else:
                pid = str(getattr(rec, "id", ""))
                payload = getattr(rec, "payload", None) or {}
            payload["_point_id"] = pid
            payloads.append(payload)
        if next_offset is None:
            break
        offset = next_offset
    return payloads


def retrieve_chunks_bm25(
    question: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    k: int = 5,
    qdrant_url: str = QDRANT_URL,
) -> List[dict]:
    """
    Klasik koleksiyondan tüm chunk'ları çekip BM25 ile sıralar.
    Dönüş: [{"text": str, "score": float}]
    """
    if not _BM25_AVAILABLE:
        raise RuntimeError("rank-bm25 paketi yüklü değil. `pip install rank-bm25` komutunu çalıştırın.")

    client = get_qdrant_client(url=qdrant_url)
    payloads = _scroll_all_payloads(client, collection_name)
    if not payloads:
        return []

    corpus_texts = [p.get("text", "") for p in payloads]
    tokenized_corpus = [_bm25_tokenize(t) for t in corpus_texts]
    bm25 = _BM25Okapi(tokenized_corpus)

    tokenized_query = _bm25_tokenize(question)
    scores = bm25.get_scores(tokenized_query)

    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

    docs: List[dict] = []
    seen: set = set()
    for idx, score in indexed:
        text = corpus_texts[idx].strip()
        if not text:
            continue
        key = _normalize_text_for_dedup(text)
        if key in seen:
            continue
        seen.add(key)
        docs.append({"text": text, "score": float(score)})
    return docs


def retrieve_chunks_bm25_smart(
    question: str,
    base_collection: str = DEFAULT_COLLECTION_NAME,
    k: int = 5,
    qdrant_url: str = QDRANT_URL,
) -> List[dict]:
    """
    Smart koleksiyonundan BM25 ile children içinde arama yapar,
    eşleşen child'ın parent_id'si ile parents koleksiyonundan tam parent bloğu çeker.
    Dönüş: [{"text": parent_text, "child_text": child_text, "score": float, "parent_id": str}]
    """
    if not _BM25_AVAILABLE:
        raise RuntimeError("rank-bm25 paketi yüklü değil. `pip install rank-bm25` komutunu çalıştırın.")

    children_col = f"{base_collection}_children"
    parents_col = f"{base_collection}_parents"

    client = get_qdrant_client(url=qdrant_url)
    child_payloads = _scroll_all_payloads(client, children_col)
    if not child_payloads:
        return []

    corpus_texts = [p.get("text", "") for p in child_payloads]
    tokenized_corpus = [_bm25_tokenize(t) for t in corpus_texts]
    bm25 = _BM25Okapi(tokenized_corpus)

    tokenized_query = _bm25_tokenize(question)
    scores = bm25.get_scores(tokenized_query)

    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

    parent_ids_ordered: List[str] = []
    child_info: dict = {}
    seen_parents: set = set()

    for idx, score in indexed:
        payload = child_payloads[idx]
        child_text = payload.get("text", "")
        parent_id = str(payload.get("parent_id", ""))
        if parent_id and parent_id not in seen_parents:
            seen_parents.add(parent_id)
            parent_ids_ordered.append(parent_id)
            child_info[parent_id] = {"child_text": child_text, "score": float(score)}

    if not parent_ids_ordered:
        return []

    try:
        parent_records = client.retrieve(
            collection_name=parents_col,
            ids=parent_ids_ordered,
            with_payload=True,
        )
    except Exception:
        return []

    parent_text_map: dict = {}
    for rec in parent_records:
        if isinstance(rec, dict):
            pid = str(rec.get("id", ""))
            payload = rec.get("payload") or {}
        else:
            pid = str(getattr(rec, "id", ""))
            payload = getattr(rec, "payload", None) or {}
        if pid:
            parent_text_map[pid] = payload.get("text", "")

    docs: List[dict] = []
    seen_text: set = set()

    for parent_id in parent_ids_ordered:
        parent_text = parent_text_map.get(parent_id, "")
        info = child_info[parent_id]
        if parent_text.strip():
            key = _normalize_text_for_dedup(parent_text)
            if key not in seen_text:
                seen_text.add(key)
                docs.append(
                    {
                        "text": parent_text,
                        "child_text": info["child_text"],
                        "score": info["score"],
                        "parent_id": parent_id,
                    }
                )

    return docs
