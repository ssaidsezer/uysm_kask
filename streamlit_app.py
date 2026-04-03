from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import requests
import streamlit as st
from dotenv import load_dotenv

WORKSPACE_DIR = Path(__file__).resolve().parent
load_dotenv(WORKSPACE_DIR / ".env", override=True)

from rag_index import (
    DEFAULT_COLLECTION_NAME,
    QDRANT_URL,
    SMART_PARENT_BLOCK_SIZE,
    SMART_CHILD_SIZE,
    SMART_CHILD_OVERLAP,
    SMART_BOUNDARY_LLM_MODEL,
    index_pdfs,
    index_pdfs_smart,
    retrieve_chunks,
    retrieve_chunks_smart,
    retrieve_chunks_bm25,
    retrieve_chunks_bm25_smart,
)
from pipeline import (
    QA_OLLAMA_MODEL,
    EVAL_MODEL_NAME,
    evaluate_answer_any,
    generate_rag_answer_ollama,
    generate_no_rag_answer_ollama,
    get_openai_client,
    run_full_pipeline,
    warmup_model,
    unload_model,
    write_results_to_csv,
)
from voice_utils import synthesize_speech, get_downloaded_tts_models


def _get_ollama_base_url() -> str:
    """Resolve Ollama base URL from env, tolerating legacy host-only config."""
    base_url = os.environ.get("OLLAMA_BASE_URL", "").strip()
    if not base_url:
        host = os.environ.get("OLLAMA_HOST", "").strip()
        if host:
            base_url = host
    if not base_url:
        return ""
    if not base_url.startswith("http"):
        base_url = f"http://{base_url}"
    return base_url.rstrip("/")


def _collection_name_full(
    base: str, embed_model: str, chunk_size: int, chunk_overlap: int
) -> str:
    """Klasik mod: model + chunk ayarına özgü tekil koleksiyon adı üretir.
    Örnek: uysm_bge-m3_latest_1000c_200ov"""
    safe = embed_model.replace(":", "_").replace("/", "_").replace(".", "_")
    return f"{base}_{safe}_{chunk_size}c_{chunk_overlap}ov"

def _smart_collection_name_full(
    base: str,
    embed_model: str,
    parent_size: int,
    child_size: int,
    child_overlap: int,
) -> str:
    """Smart mod: model + parent/child boyutuna özgü tekil koleksiyon adı üretir.
    Örnek: uysm_bge-m3_latest_3000p_500c_100ov"""
    safe = embed_model.replace(":", "_").replace("/", "_").replace(".", "_")
    return f"{base}_{safe}_{parent_size}p_{child_size}c_{child_overlap}ov"


def _collection_options_for_embed(
    collections: List[str],
    embed_model: str | None,
) -> tuple[List[str], List[str]]:
    """Return available classic collections and smart base collections for an embedding model."""
    if not embed_model:
        return [], []
    safe_embed = embed_model.replace(":", "_").replace("/", "_").replace(".", "_")
    classic_cols = sorted(
        c for c in collections
        if safe_embed in c and not c.endswith("_children") and not c.endswith("_parents")
    )
    smart_bases = sorted(
        {
            c[: -len("_children")]
            for c in collections
            if c.endswith("_children") and safe_embed in c and f"{c[: -len('_children')]}_parents" in collections
        }
    )
    return classic_cols, smart_bases

@dataclass
class RetrievalSelection:
    embed_model: str | None
    collection_name: str
    smart_rag: bool
    retrieval_mode: str


def _is_embedding_model(host: str, model_name: str) -> bool:
    """
    /api/show ile modelin embedding modeli olup olmadığını kontrol eder.
    Öncelik sırası:
      1. capabilities listesinde "embedding" varsa → kesinlikle embedding modeli
      2. Model adında "embed" veya "bge" geçiyorsa → embedding modeli
      3. template alanı boşsa → embedding modeli (eski Ollama sürümleri)
    """
    try:
        resp = requests.post(
            host.rstrip("/") + "/api/show",
            json={"name": model_name},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json() or {}
    except Exception:
        return False

    # 1. Yeni Ollama (0.4+): capabilities alanı varsa ona güven
    capabilities = data.get("capabilities") or []
    if capabilities:
        return "embedding" in capabilities

    # 2. İsim bazlı fallback
    name_lower = model_name.lower()
    if any(kw in name_lower for kw in ("embed", "bge", "e5", "gte", "rerank")):
        return True

    # 3. Template boşsa embedding modeli say (eski Ollama sürümleri)
    template = (data.get("template") or "").strip()
    if not template:
        return True

    return False

@st.cache_data(ttl=300, show_spinner=False)
def _list_ollama_models() -> tuple[List[str], str, float, int]:
    """
    List models from the remote Ollama HTTP API (/api/tags).
    Embedding modellerini /api/show ile filtreleyerek hariç tutar.
    Dönüş: (modeller, hata_mesajı, filtreleme_süresi_sn, filtrelenen_model_sayısı)
    """
    host = _get_ollama_base_url()
    if not host:
        return [], (
            "OLLAMA_BASE_URL veya OLLAMA_HOST ortam değişkeni tanımlı değil. "
            "Lütfen .env dosyasına uzak sunucu adresini ekleyin "
            "(örn: OLLAMA_BASE_URL=http://192.168.1.151:11434)."
        ), 0.0, 0

    url = host + "/api/tags"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json() or {}
    except Exception:
        return [], f"Uzak Ollama sunucusuna ({host}) bağlanılamadı. Lütfen sunucunun açık olduğundan ve ağ/güvenlik duvarı ayarlarının yapıldığından emin olun.", 0.0, 0

    all_names: List[str] = []
    for item in data.get("models", []):
        name = item.get("name")
        if isinstance(name, str):
            all_names.append(name)

    import time
    t0 = time.time()
    models: List[str] = []
    filtered_count = 0
    for name in all_names:
        if _is_embedding_model(host, name):
            filtered_count += 1
        else:
            models.append(name)
    elapsed = round(time.time() - t0, 2)

    return models, "", elapsed, filtered_count


@st.cache_data(ttl=300, show_spinner=False)
def _list_embedding_models() -> tuple[List[str], str]:
    """
    Uzak Ollama sunucusundan sadece embedding modellerini listeler.
    Dönüş: (embedding_modeller, hata_mesajı)
    """
    host = _get_ollama_base_url()
    if not host:
        return [], "OLLAMA_BASE_URL veya OLLAMA_HOST ortam değişkeni tanımlı değil."

    url = host + "/api/tags"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json() or {}
    except Exception:
        return [], f"Uzak Ollama sunucusuna ({host}) bağlanılamadı."

    all_names: List[str] = [
        item.get("name")
        for item in data.get("models", [])
        if isinstance(item.get("name"), str)
    ]

    embed_models = [name for name in all_names if _is_embedding_model(host, name)]
    return embed_models, ""


def _ensure_tmp_dir() -> Path:
    tmp_dir = WORKSPACE_DIR / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    return tmp_dir


def _pull_ollama_model(model_name: str) -> tuple[bool, str]:
    """Ollama sunucusunda model pull eder. (başarı, mesaj) döndürür."""
    host = _get_ollama_base_url()
    if not host:
        return False, "OLLAMA_BASE_URL veya OLLAMA_HOST ortam değişkeni tanımlı değil."
    url = host + "/api/pull"
    try:
        resp = requests.post(url, json={"name": model_name, "stream": False}, timeout=300)
        resp.raise_for_status()
        return True, f"'{model_name}' başarıyla pull edildi."
    except Exception as e:
        return False, f"Pull sırasında hata: {e}"


def _delete_ollama_model(model_name: str) -> tuple[bool, str]:
    """Ollama sunucusundan model siler. (başarı, mesaj) döndürür."""
    host = _get_ollama_base_url()
    if not host:
        return False, "OLLAMA_BASE_URL veya OLLAMA_HOST ortam değişkeni tanımlı değil."
    url = host + "/api/delete"
    try:
        resp = requests.delete(url, json={"name": model_name}, timeout=30)
        resp.raise_for_status()
        return True, f"'{model_name}' başarıyla silindi."
    except Exception as e:
        return False, f"Silme sırasında hata: {e}"


def _list_qdrant_collections() -> tuple[List[str], str]:
    """Qdrant'taki tüm koleksiyonları listeler. (koleksiyonlar, hata_mesajı) döndürür."""
    qdrant_url = os.environ.get("QDRANT_URL", QDRANT_URL)
    try:
        resp = requests.get(f"{qdrant_url}/collections", timeout=10)
        resp.raise_for_status()
        data = resp.json() or {}
        collections = [c["name"] for c in data.get("result", {}).get("collections", [])]
        return collections, ""
    except Exception as e:
        return [], f"Qdrant koleksiyonları listelenemedi: {e}"


def _delete_qdrant_collection(collection_name: str) -> tuple[bool, str]:
    """Qdrant'tan koleksiyon siler. (başarı, mesaj) döndürür."""
    qdrant_url = os.environ.get("QDRANT_URL", QDRANT_URL)
    try:
        resp = requests.delete(f"{qdrant_url}/collections/{collection_name}", timeout=30)
        resp.raise_for_status()
        return True, f"'{collection_name}' koleksiyonu başarıyla silindi."
    except Exception as e:
        return False, f"Koleksiyon silme sırasında hata: {e}"


def _append_chat_eval_row(
    *,
    run_timestamp: str,
    model_name: str,
    mode: str,
    question: str,
    expected_answer: str,
    record: dict,
    result: dict,
    eval_data: dict | None,
    retrieved_chunks: str,
) -> None:
    st.session_state.setdefault("chat_eval_rows", []).append(
        {
            "timestamp": run_timestamp,
            "model": model_name,
            "mode": mode,
            "question": question,
            "expected_answer": expected_answer.strip(),
            "model_answer": record["model_answer"],
            "response_time_seconds": record["response_time_seconds"],
            "eval_duration_seconds": result.get("eval_duration_seconds") or "",
            "tokens_per_second": result.get("tokens_per_second") or "",
            "ai_score": (eval_data or {}).get("ai_score", ""),
            "ai_verdict": (eval_data or {}).get("ai_verdict", ""),
            "ai_hallucination_risk": (eval_data or {}).get("ai_hallucination_risk", ""),
            "retrieved_chunks": retrieved_chunks,
        }
    )


def _run_chat_eval(
    question: str,
    expected_answer: str,
    rag_mode: str,
    k: int,
    qa_models_selected: List[str],
    all_models: List[str],
    eval_enabled: bool,
    eval_backend: str,
    eval_model_name: str,
    local_eval_model_name: str | None,
    openai_api_key: str,
    collection_name: str,
    embed_model: str | None = None,
    think: bool = False,
    smart_rag: bool = False,
    score_threshold: float = 0.55,
    retrieval_mode: str = "vector",
) -> List[dict]:
    """Run RAG (and optionally no-RAG) QA + evaluation for given models.

    Displays results via Streamlit and appends to session state.
    Returns list of result dicts containing 'model_answer' for TTS.
    smart_rag=True: child chunk eşleşmesi yapılır, modele parent bağlamı verilir.
    retrieval_mode: 'vector' | 'bm25'
    """
    if eval_enabled and eval_backend == "OpenAI":
        if not openai_api_key and not os.environ.get("OPENAI_API_KEY"):
            st.error(
                "OpenAI değerlendirme motoru seçili. OpenAI API key gerekli "
                "(sidebar'dan girin veya ortam değişkeni ayarlayın)."
            )
            return []
        openai_client = get_openai_client(api_key=openai_api_key or None)
    else:
        openai_client = None

    retrieved_chunks_list: List[dict] = []
    context = ""
    if rag_mode in ("rag", "both"):
        if retrieval_mode == "bm25":
            if smart_rag:
                retrieved_chunks_list = retrieve_chunks_bm25_smart(
                    question=question,
                    base_collection=collection_name,
                    k=int(k),
                )
            else:
                retrieved_chunks_list = retrieve_chunks_bm25(
                    question=question,
                    collection_name=collection_name,
                    k=int(k),
                )
        elif smart_rag:
            retrieved_chunks_list = retrieve_chunks_smart(
                question=question,
                base_collection=collection_name,
                k=int(k),
                embed_model=embed_model or None,
                score_threshold=score_threshold,
            )
        else:
            retrieved_chunks_list = retrieve_chunks(
                question=question,
                collection_name=collection_name,
                k=int(k),
                embed_model=embed_model or None,
                score_threshold=score_threshold,
            )
        context = "\n\n".join(c["text"] for c in retrieved_chunks_list)

    # --- Chunk kartları (modelden bağımsız, bir kez göster) ---
    if rag_mode in ("rag", "both") and retrieved_chunks_list:
        if retrieval_mode == "bm25" and smart_rag:
            st.markdown("**Retrieved Chunks — BM25 Smart** *(BM25 ile child arama, bağlam olarak parent blok kullanılır)*")
        elif retrieval_mode == "bm25":
            st.markdown("**Retrieved Chunks — BM25 Klasik** *(anahtar kelime tabanlı BM25 arama)*")
        elif smart_rag:
            st.markdown("**Retrieved Chunks — Smart RAG** *(kartlar eşleşen child chunk'ı, bağlam olarak parent blok kullanılır)*")
        else:
            st.markdown("**Retrieved Chunks**")
        chunk_cols = st.columns(4)
        for i, chunk in enumerate(retrieved_chunks_list):
            score = chunk["score"]
            score_color = "#4caf50" if score >= 0.6 else "#ff9800" if score >= 0.4 else "#f44336"
            child_preview = chunk.get("child_text", chunk["text"])
            parent_preview = chunk["text"]
            with chunk_cols[i % 4]:
                if smart_rag:
                    st.markdown(
                        f"""<div style="border:1px solid #4a90d9;border-radius:8px;padding:10px;margin-bottom:8px;font-size:0.78rem;background:#1e1e1e;position:relative;">
                        <b style="color:#4a90d9;">Chunk {i + 1} — Child eşleşmesi</b><br>
                        <span style="color:#aaa;font-size:0.72rem;">↓ eşleşen child</span><br>
                        <div style="max-height:60px;overflow-y:auto;color:#ccc;">{child_preview}</div>
                        <hr style="border-color:#333;margin:6px 0;">
                        <span style="color:#aaa;font-size:0.72rem;">↓ modele verilen parent bağlam</span><br>
                        <div style="max-height:80px;overflow-y:auto;color:#eee;">{parent_preview}</div>
                        <div style="text-align:right;margin-top:6px;">
                            <span style="background:#2a2a2a;border-radius:4px;padding:2px 6px;font-size:0.72rem;color:{score_color};font-weight:bold;">
                                {score:.3f}
                            </span>
                        </div></div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""<div style="border:1px solid #444;border-radius:8px;padding:10px;margin-bottom:8px;font-size:0.78rem;height:160px;overflow-y:auto;background:#1e1e1e;position:relative;">
                        <b style="color:#aaa;">Chunk {i + 1}</b><br><br>{parent_preview}
                        <div style="position:sticky;bottom:0;text-align:right;margin-top:6px;">
                            <span style="background:#2a2a2a;border-radius:4px;padding:2px 6px;font-size:0.72rem;color:{score_color};font-weight:bold;">
                                {score:.3f}
                            </span>
                        </div></div>""",
                        unsafe_allow_html=True,
                    )
    elif rag_mode in ("rag", "both"):
        if retrieval_mode == "bm25":
            st.caption("BM25 araması herhangi bir chunk döndürmedi. Koleksiyonun dolu olduğundan emin olun.")
        else:
            st.caption(
                f"Qdrant'ta bu sorgu için `score_threshold={score_threshold:.2f}` üstünde chunk bulunamadı. "
                "Daha fazla sonuç için threshold değerini düşürmeyi deneyin."
            )

    st.divider()

    run_timestamp = datetime.utcnow().isoformat()
    selected_models = qa_models_selected or all_models
    answers: List[dict] = []

    for qa_model_name in selected_models:
        warmup_model(model=qa_model_name)

        st.markdown(f"### {qa_model_name}")

        rag_record = rag_result = rag_eval = None
        no_rag_record = no_rag_result = no_rag_eval = None

        # --- RAG'li cevap üret ---
        if rag_mode in ("rag", "both"):
            try:
                with st.spinner(f"{qa_model_name} — RAG'li cevap üretiliyor..."):
                    rag_result = generate_rag_answer_ollama(
                        question=question,
                        context=context,
                        model=qa_model_name,
                        think=think,
                    )
                    rag_record = {
                        "model": f"{qa_model_name} (RAG)" if rag_mode == "both" else qa_model_name,
                        "question_index": 0,
                        "question": question,
                        "observation_idea": expected_answer or "",
                        "model_answer": rag_result.get("answer", ""),
                        "response_time_seconds": rag_result.get("response_time_seconds", 0.0),
                    }
                    if eval_enabled:
                        rag_eval = evaluate_answer_any(
                            record=rag_record,
                            eval_model=eval_model_name,
                            client=openai_client,
                            backend="openai" if eval_backend == "OpenAI" else "ollama",
                            local_model=local_eval_model_name,
                        )
                    else:
                        rag_eval = {}
            except Exception as exc:
                st.error(f"{qa_model_name} için RAG'li çağrıda hata oluştu ve model atlandı: {exc}")
                continue

        # --- RAG'siz cevap üret ---
        if rag_mode in ("no_rag", "both"):
            try:
                with st.spinner(f"{qa_model_name} — RAG'siz cevap üretiliyor..."):
                    no_rag_result = generate_no_rag_answer_ollama(
                        question=question,
                        model=qa_model_name,
                        think=think,
                    )
                    no_rag_record = {
                        "model": f"{qa_model_name} (RAG'siz)" if rag_mode == "both" else qa_model_name,
                        "question_index": 0,
                        "question": question,
                        "observation_idea": expected_answer or "",
                        "model_answer": no_rag_result.get("answer", ""),
                        "response_time_seconds": no_rag_result.get("response_time_seconds", 0.0),
                    }
                    if eval_enabled:
                        no_rag_eval = evaluate_answer_any(
                            record=no_rag_record,
                            eval_model=eval_model_name,
                            client=openai_client,
                            backend="openai" if eval_backend == "OpenAI" else "ollama",
                            local_model=local_eval_model_name,
                        )
                    else:
                        no_rag_eval = {}
            except Exception as exc:
                st.error(f"{qa_model_name} için RAG'siz çağrıda hata oluştu ve bu mod atlandı: {exc}")
                continue

        # --- Her model için hizalanmış kolon render ---
        if rag_mode == "both":
            if not rag_record or not no_rag_record or not rag_result or not no_rag_result:
                st.error(
                    f"{qa_model_name} için çift mod çıktısı eksik. "
                    "RAG veya RAG'siz çağrılardan biri tamamlanamadı."
                )
                unload_model(qa_model_name)
                st.divider()
                continue

            rag_col, no_rag_col = st.columns(2)
            with rag_col:
                st.markdown("**RAG'li**")
                st.markdown(rag_record["model_answer"])
                if eval_enabled:
                    with st.expander("Eval"):
                        st.json(rag_eval)
            with no_rag_col:
                st.markdown("**RAG'siz**")
                st.markdown(no_rag_record["model_answer"])
                if eval_enabled:
                    with st.expander("Eval"):
                        st.json(no_rag_eval)
        elif rag_mode == "rag" and rag_record:
            st.markdown(rag_record["model_answer"])
            if eval_enabled:
                with st.expander("Eval"):
                    st.json(rag_eval)
        elif rag_mode == "no_rag" and no_rag_record:
            st.markdown(no_rag_record["model_answer"])
            if eval_enabled:
                with st.expander("Eval"):
                    st.json(no_rag_eval)

        # --- Session state kayıt ---
        if rag_record and rag_result:
            st.session_state.setdefault("chat_eval_rows", []).append(
                {
                    "timestamp": run_timestamp,
                    "model": qa_model_name,
                    "mode": "RAG",
                    "question": question,
                    "expected_answer": (expected_answer or "").strip(),
                    "model_answer": rag_record["model_answer"],
                    "response_time_seconds": rag_record["response_time_seconds"],
                    "tokens_per_second": rag_result.get("tokens_per_second") or "",
                    "ai_score": (rag_eval or {}).get("ai_score", ""),
                    "ai_verdict": (rag_eval or {}).get("ai_verdict", ""),
                    "ai_hallucination_risk": (rag_eval or {}).get("ai_hallucination_risk", ""),
                    "retrieved_chunks": json.dumps([c["text"] for c in retrieved_chunks_list], ensure_ascii=False),
                }
            )
            answers.append({"model": qa_model_name, "mode": "RAG", "answer": rag_record["model_answer"]})

        if no_rag_record and no_rag_result:
            _append_chat_eval_row(
                run_timestamp=run_timestamp,
                model_name=qa_model_name,
                mode="NO_RAG",
                question=question,
                expected_answer=expected_answer or "",
                record=no_rag_record,
                result=no_rag_result,
                eval_data=no_rag_eval,
                retrieved_chunks="[]",
            )
            answers.append({"model": qa_model_name, "mode": "NO_RAG", "answer": no_rag_record["model_answer"]})

        unload_model(qa_model_name)
        st.divider()

    return answers


@st.fragment
def _render_qa_model_selector(all_models: List[str], filtered_count: int, key_prefix: str) -> None:
    """QA model seçim UI'ı render eder, seçili modelleri session_state'e yazar."""
    search_key = f"{key_prefix}_qa_model_search"
    filtered_key = f"_{key_prefix}_qa_filtered_models"
    custom_models_key = f"{key_prefix}_custom_models"

    if custom_models_key not in st.session_state:
        st.session_state[custom_models_key] = []

    # Custom modelleri listenin başına ekle
    combined_models = list(all_models)
    for cm in st.session_state[custom_models_key]:
        if cm not in combined_models:
            combined_models.insert(0, cm)

    search_value = st.session_state.get(search_key, "")
    filtered_models = (
        [m for m in combined_models if search_value.lower() in m.lower()] if search_value else combined_models
    )

    def _select_all():
        for m in st.session_state.get(filtered_key, []):
            st.session_state[f"{key_prefix}_qa_model_select_{m}"] = True

    def _deselect_all():
        for m in st.session_state.get(filtered_key, []):
            st.session_state[f"{key_prefix}_qa_model_select_{m}"] = False

    st.session_state[filtered_key] = filtered_models

    selected_count = sum(
        1 for m in combined_models if st.session_state.get(f"{key_prefix}_qa_model_select_{m}", False)
    )
    expander_label = f"Modelleri göster ({selected_count}/{len(combined_models)} seçili)"
    if filtered_count > 0:
        expander_label += f" · {filtered_count} embedding filtrelendi"

    with st.expander(expander_label, expanded=True):
        col_search, col_sel, col_desel = st.columns([6, 1, 1])
        with col_search:
            st.text_input("Model ara", placeholder="Model adında ara...", key=search_key, label_visibility="collapsed")
        with col_sel:
            st.button("Hepsini seç", key=f"{key_prefix}_qa_select_all", on_click=_select_all, use_container_width=True)
        with col_desel:
            st.button("Hepsini kaldır", key=f"{key_prefix}_qa_deselect_all", on_click=_deselect_all, use_container_width=True)
        grid_cols = st.columns(3)
        for i, model_name in enumerate(filtered_models):
            with grid_cols[i % 3]:
                st.checkbox(
                    model_name,
                    value=st.session_state.get(f"{key_prefix}_qa_model_select_{model_name}", False),
                    key=f"{key_prefix}_qa_model_select_{model_name}",
                    help="Bu modeli RAG değerlendirmesine dahil et.",
                )

    # Seçili modelleri session_state'e yaz (ana script okuyabilsin)
    st.session_state[f"{key_prefix}_qa_models_selected"] = [
        m for m in combined_models if st.session_state.get(f"{key_prefix}_qa_model_select_{m}", False)
    ]

def _render_eval_settings(all_models: List[str], key_prefix: str):
    """Değerlendirme motoru ayarlarını render eder. (eval_enabled, eval_backend, eval_model_name, local_eval_model_name) döndürür."""
    col_toggle, col_backend, col_model = st.columns([1, 2, 2])
    with col_toggle:
        eval_enabled = st.toggle(
            "Değerlendir",
            value=True,
            key=f"{key_prefix}_eval_enabled",
            help="Kapalıysa cevaplar üretilir fakat AI değerlendirmesi yapılmaz.",
        )
    if not eval_enabled:
        return False, "OpenAI", EVAL_MODEL_NAME, None
    with col_backend:
        eval_backend = st.selectbox(
            "Değerlendirme motoru",
            options=["OpenAI", "Yerel (Ollama)"],
            index=0,
            key=f"{key_prefix}_eval_backend",
            help="Cevapları OpenAI ile mi yoksa yerel bir Ollama modeliyle mi değerlendireceğini seç.",
        )
    local_eval_model_name: str | None = None
    with col_model:
        if eval_backend == "OpenAI":
            eval_model_name = st.text_input(
                "OpenAI değerlendirme modeli",
                value=EVAL_MODEL_NAME,
                key=f"{key_prefix}_eval_model_name",
                help="OpenAI değerlendirme motoru seçiliyse kullanılacak model.",
            )
        else:
            eval_model_name = EVAL_MODEL_NAME
            st.empty()
    if eval_backend == "Yerel (Ollama)":
        local_eval_model_name = st.selectbox(
            "Yerel değerlendirme modeli (Ollama)",
            options=all_models if all_models else ["Bağlantı hatası/Model Yok"],
            index=0,
            key=f"{key_prefix}_local_eval_model",
            help="Eval için kullanılacak yerel Ollama modelini seç.",
        )
    return True, eval_backend, eval_model_name, local_eval_model_name


def _render_collection_selection(
    *,
    shared_embed_models: List[str],
    collection_name: str,
    key_prefix: str,
    embed_label: str,
    rag_type_label: str,
    rag_type_help: str,
    rag_type_horizontal: bool,
    classic_label: str,
    smart_label: str,
    classic_help: str,
    smart_help: str,
    classic_caption_prefix: str,
    smart_caption_prefix: str,
) -> RetrievalSelection:
    collections, coll_err = _list_qdrant_collections()
    if coll_err:
        st.warning(coll_err)
        collections = []

    if not shared_embed_models:
        return RetrievalSelection(
            embed_model=None,
            collection_name=collection_name,
            smart_rag=False,
            retrieval_mode="vector",
        )

    embed_model = st.selectbox(
        embed_label,
        options=shared_embed_models,
        key=f"{key_prefix}_embed_model",
    )
    classic_cols, smart_cols = _collection_options_for_embed(collections, embed_model)
    rag_type = st.radio(
        rag_type_label,
        options=["Klasik", "Smart", "BM25 Klasik", "BM25 Smart"],
        horizontal=rag_type_horizontal,
        key=f"{key_prefix}_rag_type",
        help=rag_type_help,
    )
    smart_rag = rag_type in ("Smart", "BM25 Smart")
    retrieval_mode = "bm25" if rag_type.startswith("BM25") else "vector"

    if smart_rag:
        if smart_cols:
            selected_collection = st.selectbox(
                smart_label,
                options=smart_cols,
                key=f"{key_prefix}_smart_col_select",
                help=smart_help,
            )
            st.caption(
                f"{smart_caption_prefix}: **{selected_collection}_children** / **{selected_collection}_parents**"
            )
        else:
            selected_collection = ""
            st.warning(f"'{embed_model}' modeli için smart koleksiyon bulunamadı.")
    else:
        if classic_cols:
            selected_collection = st.selectbox(
                classic_label,
                options=classic_cols,
                key=f"{key_prefix}_classic_col_select",
                help=classic_help,
            )
            st.caption(f"{classic_caption_prefix}: **{selected_collection}**")
        else:
            selected_collection = ""
            st.warning(f"'{embed_model}' modeli için klasik koleksiyon bulunamadı.")

    return RetrievalSelection(
        embed_model=embed_model,
        collection_name=selected_collection,
        smart_rag=smart_rag,
        retrieval_mode=retrieval_mode,
    )


@st.fragment
def _render_csv_eval_tab(
    connection_error: str | None,
    all_models: List[str],
    filtered_count: int,
    shared_embed_models: List[str],
    collection_name: str,
) -> None:
    st.subheader("CSV'den soruları değerlendir")

    st.markdown("**Değerlendirilecek QA modelleri**")
    if connection_error:
        st.error(connection_error)
    _render_qa_model_selector(all_models, filtered_count, key_prefix="csv")
    qa_models_selected = st.session_state.get("csv_qa_models_selected", [])

    st.markdown("---")
    eval_enabled, eval_backend, eval_model_name, local_eval_model_name = _render_eval_settings(all_models, key_prefix="csv")
    st.markdown("---")

    uploaded_csv = st.file_uploader("CSV yükle", type=["csv"])

    col_qcol, col_acol = st.columns(2)
    with col_qcol:
        csv_question_col = st.text_input(
            "Soruların bulunduğu sütun adı",
            value="question",
            key="csv_question_col",
        )
    with col_acol:
        csv_answer_col = st.text_input(
            "Cevapların bulunduğu sütun adı",
            value="answer",
            key="csv_answer_col",
        )

    sample_csv_path = WORKSPACE_DIR / "sample_rag_input.csv"
    use_sample = False
    if sample_csv_path.exists():
        use_sample = st.checkbox(
            "Varsayılan örnek CSV'yi kullan (sample_rag_input.csv)",
            value=not uploaded_csv,
        )

    retrieval_selection = _render_collection_selection(
        shared_embed_models=shared_embed_models,
        collection_name=collection_name,
        key_prefix="csv",
        embed_label="Embedding modeli (indexleme ile aynı olmalı)",
        rag_type_label="RAG modu (indeksleme ile aynı olmalı)",
        rag_type_help="BM25 anahtar kelime tabanlıdır. Smart mod parent/child koleksiyonlarını kullanır.",
        rag_type_horizontal=True,
        classic_label="Klasik koleksiyon",
        smart_label="Smart koleksiyon",
        classic_help="İndeksleme tabında oluşturulan klasik koleksiyonu seçin.",
        smart_help="İndeksleme tabında oluşturulan smart base koleksiyonu seçin.",
        classic_caption_prefix="Kullanılacak koleksiyon",
        smart_caption_prefix="Kullanılacak koleksiyonlar",
    )
    csv_embed_model = retrieval_selection.embed_model
    csv_collection_name = retrieval_selection.collection_name
    csv_smart_rag = retrieval_selection.smart_rag
    csv_retrieval_mode = retrieval_selection.retrieval_mode
    csv_score_threshold = 0.55

    col_k, col_rag_mode, col_think = st.columns([2, 2, 1])
    with col_rag_mode:
        rag_mode_label = st.radio(
            "Cevaplama modu",
            options=["RAG'li", "RAG'siz", "İkisi birden"],
            horizontal=True,
            key="csv_rag_mode",
        )
    rag_mode_map = {"RAG'li": "rag", "RAG'siz": "no_rag", "İkisi birden": "both"}
    rag_mode = rag_mode_map[rag_mode_label]
    with col_k:
        if rag_mode != "no_rag":
            k = st.number_input(
                "Her soru için alınacak context chunk sayısı (k)",
                min_value=1,
                max_value=20,
                value=5,
            )
        else:
            k = 5
    with col_think:
        thinking_enabled = st.toggle(
            "Thinking modu",
            value=False,
            key="csv_thinking_enabled",
            help=(
                "Reasoning modellerinde (Qwen3, QwQ vb.) thinking modunu açar. "
                "Kapalıyken <think> blokları cevaba karışmaz, eval sonuçları temiz kalır. "
                "Diğer modeller bu seçeneği zaten görmezden gelir."
            ),
        )

    if rag_mode != "no_rag" and csv_retrieval_mode == "vector":
        csv_score_threshold = st.slider(
            "Minimum eşleşme skoru (score threshold)",
            min_value=0.10,
            max_value=1.0,
            value=0.55,
            step=0.05,
            key="csv_score_threshold",
            help="Düşük değer daha fazla ama daha zayıf eşleşme döndürür.",
        )

    qdrant_url = os.environ.get("QDRANT_URL", QDRANT_URL)
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")

    # Akış:
    # 1. Buton tıklanır → eski sonuçlar silinir, kwargs session_state'e yazılır, fragment rerun.
    # 2. Sonraki render'da kwargs bulunur → pipeline çalışır → sonuçlar session_state'e yazılır.
    # 3. Sonuçlar her zaman buton bloğunun dışında session_state'den render edilir.
    # Bu sayede pipeline çalışırken eski download butonu görünmez.

    if st.button("Pipeline'ı çalıştır"):
        if rag_mode != "no_rag" and not csv_collection_name:
            st.error("Seçili retrieval tipi için geçerli bir koleksiyon bulunamadı.")
        else:
            st.session_state.csv_results = None
            st.session_state.csv_errors = []
            st.session_state._csv_run_kwargs = dict(
                uploaded_csv_bytes=uploaded_csv.getbuffer().tobytes() if uploaded_csv is not None else None,
                use_sample=use_sample,
                sample_csv_path=str(sample_csv_path),
                eval_enabled=eval_enabled,
                eval_backend=eval_backend,
                eval_model_name=eval_model_name,
                local_eval_model_name=local_eval_model_name,
                csv_question_col=csv_question_col,
                csv_answer_col=csv_answer_col,
                csv_embed_model=csv_embed_model,
                csv_collection_name=csv_collection_name,
                csv_smart_rag=csv_smart_rag,
                csv_retrieval_mode=csv_retrieval_mode,
                csv_score_threshold=float(csv_score_threshold),
                rag_mode=rag_mode,
                rag_mode_label=rag_mode_label,
                k=int(k),
                thinking_enabled=thinking_enabled,
                qa_models_selected=qa_models_selected,
                qdrant_url=qdrant_url,
                openai_api_key=openai_api_key,
            )
            st.rerun(scope="fragment")

    _kwargs = st.session_state.pop("_csv_run_kwargs", None)
    if _kwargs is not None:
        rows: list = []
        errors: list = []
        _sample_path = Path(_kwargs["sample_csv_path"])

        csv_path = None
        if _kwargs["uploaded_csv_bytes"] is not None:
            tmp_dir = _ensure_tmp_dir()
            csv_path = tmp_dir / "uploaded_input.csv"
            csv_path.write_bytes(_kwargs["uploaded_csv_bytes"])
        elif _kwargs["use_sample"] and _sample_path.exists():
            csv_path = _sample_path
        else:
            errors.append("CSV seçilmedi.")

        client = None
        if csv_path is not None and _kwargs["eval_enabled"] and _kwargs["eval_backend"] == "OpenAI":
            _api_key = _kwargs["openai_api_key"] or os.environ.get("OPENAI_API_KEY", "")
            if not _api_key:
                errors.append("OpenAI değerlendirme motoru seçili. OpenAI API key gerekli.")
                csv_path = None
            else:
                client = get_openai_client(api_key=_api_key)

        if csv_path is not None:
            _models = _kwargs["qa_models_selected"] or ([QA_OLLAMA_MODEL] if not all_models else [all_models[0]])
            for qa_model in _models:
                try:
                    with st.spinner(f"Pipeline çalışıyor: {qa_model} ({_kwargs['rag_mode_label']})..."):
                        model_rows = run_full_pipeline(
                            csv_path=str(csv_path),
                            collection_name=_kwargs["csv_collection_name"],
                            qdrant_url=_kwargs["qdrant_url"],
                            eval_model=_kwargs["eval_model_name"],
                            k=_kwargs["k"],
                            openai_client=client,
                            eval_backend="openai" if _kwargs["eval_backend"] == "OpenAI" else "ollama",
                            eval_local_model=_kwargs["local_eval_model_name"],
                            qa_model=qa_model,
                            rag_mode=_kwargs["rag_mode"],
                            eval_enabled=_kwargs["eval_enabled"],
                            question_col=_kwargs["csv_question_col"],
                            answer_col=_kwargs["csv_answer_col"],
                            embed_model=_kwargs["csv_embed_model"],
                            think=_kwargs.get("thinking_enabled", False),
                            smart_chunking=_kwargs.get("csv_smart_rag", False),
                            score_threshold=float(_kwargs.get("csv_score_threshold", 0.55)),
                            retrieval_mode=_kwargs.get("csv_retrieval_mode", "vector"),
                        )
                        rows.extend(model_rows)
                except Exception as exc:
                    errors.append(f"{qa_model} için pipeline çalışırken hata oluştu: {exc}")
                finally:
                    with st.spinner(f"{qa_model} VRAM'den boşaltılıyor..."):
                        unload_model(qa_model)

        st.session_state.csv_results = rows
        st.session_state.csv_errors = errors

    # Sonuçları session_state'den render et.
    for err in st.session_state.get("csv_errors", []):
        st.error(err)

    _rows = st.session_state.get("csv_results")
    if _rows is not None:
        if not _rows:
            st.warning("Hiç satır üretilmedi.")
        else:
            st.success(f"Pipeline tamamlandı. Toplam {len(_rows)} satır üretildi.")
            st.dataframe(_rows)
            output_csv = io.StringIO()
            _ = write_results_to_csv(_rows, output_path=output_csv)
            st.download_button(
                label="Sonuç CSV'yi indir",
                data=output_csv.getvalue().encode("utf-8"),
                file_name="output.csv",
                mime="text/csv",
            )


def _render_sidebar_monitor() -> None:
    with st.sidebar:
        st.subheader("🧠 Sunucu Kaynakları")
        monitor_url = os.environ.get("MONITOR_URL", "http://192.168.1.151:8081")

        st.caption(f"Hedef: {monitor_url}")
        if st.button("🔄 Kaynakları Yenile"):
            pass

        try:
            m_resp = requests.get(f"{monitor_url}/stats", timeout=3)
            if m_resp.status_code == 200:
                stats = m_resp.json()

                cpu_val = stats.get("cpu_usage", 0.0)
                gpu_val = stats.get("gpu_usage", 0.0)
                vram_u = stats.get("vram_used", 0)
                vram_t = stats.get("vram_total", 0)

                c1, c2 = st.columns(2)
                c1.metric("CPU", f"%{cpu_val:.1f}")
                c2.metric("GPU", f"%{gpu_val:.1f}")

                if vram_t > 0:
                    vram_pct = (float(vram_u) / float(vram_t)) * 100
                    st.metric("VRAM", f"{vram_u} / {vram_t} MB", f"%{vram_pct:.1f} dolu", delta_color="off")
                else:
                    st.metric("VRAM", f"{vram_u} MB", "Bilgi alınamadı", delta_color="off")
            else:
                st.error("Sistem bilgisi alınamadı.")
        except Exception:
            st.error("Monitor servisine ulaşılamadı. Lütfen sunucudaki Docker'ın açık olduğundan ve port 8081'in açık olduğundan emin olun.")


def _render_connection_status(qdrant_url: str) -> None:
    conn_cols = st.columns(2)
    with conn_cols[0]:
        try:
            resp = requests.get(f"{qdrant_url}/collections", timeout=5)
            resp.raise_for_status()
            st.success(f"Qdrant bağlı ({qdrant_url})")
        except Exception:
            st.error(f"Qdrant erişilemiyor ({qdrant_url})")

    ollama_base = _get_ollama_base_url()
    if ollama_base:
        with conn_cols[1]:
            try:
                resp = requests.get(f"{ollama_base}/api/tags", timeout=5)
                resp.raise_for_status()
                st.success(f"Ollama bağlı ({ollama_base})")
            except Exception:
                st.error(f"Ollama erişilemiyor ({ollama_base})")

def _render_index_tab(
    all_models: List[str],
    shared_embed_models: List[str],
    collection_name: str,
    qdrant_url: str,
) -> None:
    st.subheader("PDF'leri Qdrant'a indeksle (Ollama Embedding)")

    uploaded_pdfs = st.file_uploader(
        "PDF yükle",
        type=["pdf"],
        accept_multiple_files=True,
    )

    index_mode = st.radio(
        "İndeksleme modu",
        options=["Klasik (sabit boyut chunking)", "Smart (LLM semantik chunking)"],
        horizontal=True,
        key="index_mode",
    )
    use_smart_index = index_mode.startswith("Smart")

    chunk_size = int(os.environ.get("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", "200"))
    smart_parent_size = SMART_PARENT_BLOCK_SIZE
    smart_child_size = SMART_CHILD_SIZE
    smart_child_overlap = SMART_CHILD_OVERLAP
    smart_boundary_model = SMART_BOUNDARY_LLM_MODEL

    if shared_embed_models:
        default_embed = os.environ.get("OLLAMA_EMBED_MODEL", "")
        default_index = shared_embed_models.index(default_embed) if default_embed in shared_embed_models else 0
        embed_model_name = st.selectbox(
            "Embedding modeli",
            options=shared_embed_models,
            index=default_index,
            help="PDF'leri indekslemek için kullanılacak Ollama embedding modeli.",
        )

        if use_smart_index:
            with st.expander("Smart Chunking Parametreleri", expanded=True):
                sc_col1, sc_col2, sc_col3 = st.columns(3)
                with sc_col1:
                    smart_parent_size = st.number_input(
                        "Parent blok boyutu (karakter)",
                        min_value=500,
                        max_value=10000,
                        value=SMART_PARENT_BLOCK_SIZE,
                        step=500,
                        key="smart_parent_size",
                        help="İlk ham bölümleme için büyük blok boyutu.",
                    )
                with sc_col2:
                    smart_child_size = st.number_input(
                        "Child chunk boyutu (karakter)",
                        min_value=100,
                        max_value=3000,
                        value=SMART_CHILD_SIZE,
                        step=100,
                        key="smart_child_size",
                        help="Embedding için küçük child chunk boyutu.",
                    )
                with sc_col3:
                    smart_child_overlap = st.number_input(
                        "Child overlap (karakter)",
                        min_value=0,
                        max_value=500,
                        value=SMART_CHILD_OVERLAP,
                        step=50,
                        key="smart_child_overlap",
                    )
                boundary_llm_opts = all_models if all_models else []
                boundary_default = (
                    boundary_llm_opts.index(SMART_BOUNDARY_LLM_MODEL)
                    if SMART_BOUNDARY_LLM_MODEL in boundary_llm_opts
                    else 0
                )
                smart_boundary_model = st.selectbox(
                    "Sınır tespiti LLM modeli",
                    options=boundary_llm_opts if boundary_llm_opts else ["Model Yok"],
                    index=boundary_default if boundary_llm_opts else 0,
                    key="smart_boundary_model",
                    help="Semantik sınırları belirlemek için kullanılacak Ollama modeli.",
                )

            index_collection_name = _smart_collection_name_full(
                collection_name,
                embed_model_name,
                int(smart_parent_size),
                int(smart_child_size),
                int(smart_child_overlap),
            )
            st.info(
                f"Smart RAG | Embedding: **{embed_model_name}** | "
                f"Sınır LLM: **{smart_boundary_model}** | "
                f"Koleksiyonlar: **{index_collection_name}_parents** / **{index_collection_name}_children** | "
                f"Qdrant: **{qdrant_url}**"
            )
            if int(smart_child_size) >= int(smart_parent_size):
                st.warning(
                    f"Child chunk boyutu ({smart_child_size}) parent boyutuna ({smart_parent_size}) eşit veya büyük. "
                    "Her parent tek bir child olacak; Smart chunking'in hassas vektör arama avantajı azalır."
                )
        else:
            cl_col1, cl_col2 = st.columns(2)
            with cl_col1:
                chunk_size = st.number_input(
                    "Chunk boyutu (karakter)",
                    min_value=100,
                    max_value=5000,
                    value=chunk_size,
                    step=100,
                    key="classic_chunk_size",
                    help="Her chunk'ın maksimum karakter sayısı.",
                )
            with cl_col2:
                chunk_overlap = st.number_input(
                    "Chunk overlap (karakter)",
                    min_value=0,
                    max_value=1000,
                    value=chunk_overlap,
                    step=50,
                    key="classic_chunk_overlap",
                    help="Ardışık chunk'lar arasındaki örtüşme miktarı.",
                )
            index_collection_name = _collection_name_full(
                collection_name,
                embed_model_name,
                int(chunk_size),
                int(chunk_overlap),
            )
            st.info(
                f"Klasik RAG | Embedding: **{embed_model_name}** | "
                f"Koleksiyon: **{index_collection_name}** | Qdrant: **{qdrant_url}**"
            )

    if shared_embed_models and st.button("İndeksi oluştur / güncelle"):
        pdf_paths: List[Path] = []

        if uploaded_pdfs:
            tmp_dir = _ensure_tmp_dir()
            for up in uploaded_pdfs:
                tmp_path = tmp_dir / up.name
                with tmp_path.open("wb") as f:
                    f.write(up.getbuffer())
                pdf_paths.append(tmp_path)

        if not pdf_paths:
            st.error("İndekslenecek PDF bulunamadı.")
        else:
            try:
                progress_bar = st.progress(0, text="Hazırlanıyor...")
                status_text = st.empty()

                if use_smart_index:
                    phase_labels = {
                        "pdf_extract": "PDF'ler okunuyor",
                        "llm_boundary": "LLM semantik sınırlar belirleniyor",
                        "parent_store": "Parent bloklar Qdrant'a yazılıyor",
                        "child_embed": "Child embedding hesaplanıyor",
                        "child_upsert": "Child chunk'lar Qdrant'a yazılıyor",
                    }
                    phase_weights = {
                        "pdf_extract": 0.08,
                        "llm_boundary": 0.42,
                        "parent_store": 0.10,
                        "child_embed": 0.25,
                        "child_upsert": 0.15,
                    }
                    phase_starts = {
                        "pdf_extract": 0.0,
                        "llm_boundary": 0.08,
                        "parent_store": 0.50,
                        "child_embed": 0.60,
                        "child_upsert": 0.85,
                    }
                else:
                    phase_labels = {
                        "pdf_extract": "PDF'ler okunuyor",
                        "ollama_embed": "Ollama embedding hesaplanıyor",
                        "qdrant_upsert": "Qdrant'a yazılıyor",
                    }
                    phase_weights = {
                        "pdf_extract": 0.10,
                        "ollama_embed": 0.70,
                        "qdrant_upsert": 0.20,
                    }
                    phase_starts = {
                        "pdf_extract": 0.0,
                        "ollama_embed": 0.10,
                        "qdrant_upsert": 0.80,
                    }

                def on_progress(phase, current, total, elapsed_sec):
                    if total <= 0:
                        return
                    label = phase_labels.get(phase, phase)
                    pct_in_phase = current / total
                    overall = phase_starts.get(phase, 0.0) + phase_weights.get(phase, 0.0) * pct_in_phase
                    progress_bar.progress(min(overall, 1.0), text=f"{label}  ({current}/{total})  {elapsed_sec:.1f}s")
                    status_text.caption(f"{label}: {current}/{total} — {elapsed_sec:.1f} saniye")

                if use_smart_index:
                    result = index_pdfs_smart(
                        [str(p) for p in pdf_paths],
                        base_collection=index_collection_name,
                        parent_size=int(smart_parent_size),
                        child_size=int(smart_child_size),
                        child_overlap=int(smart_child_overlap),
                        boundary_llm_model=smart_boundary_model,
                        qdrant_url=qdrant_url,
                        embed_model=embed_model_name,
                        progress_callback=on_progress,
                    )
                else:
                    result = index_pdfs(
                        [str(p) for p in pdf_paths],
                        collection_name=index_collection_name,
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                        qdrant_url=qdrant_url,
                        embed_model=embed_model_name,
                        progress_callback=on_progress,
                    )

                progress_bar.progress(1.0, text="Tamamlandı!")
                status_text.empty()

                if use_smart_index:
                    st.success(
                        "Smart indeksleme tamamlandı! "
                        f"Toplam **{result['total_parents']}** parent ve **{result['total_children']}** child yazıldı."
                    )
                    st.markdown("#### Süre Detayları")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("PDF Okuma", f"{result['pdf_extract_sec']}s")
                    col2.metric("LLM Boundary", f"{result['llm_boundary_sec']}s")
                    col3.metric("Parent Store", f"{result['parent_store_sec']}s")
                    col4, col5, col6 = st.columns(3)
                    col4.metric("Child Embed", f"{result['child_embed_sec']}s")
                    col5.metric("Child Upsert", f"{result['child_upsert_sec']}s")
                    col6.metric("Toplam", f"{result['total_sec']}s")
                    if result.get("timeout_count"):
                        st.warning(f"LLM boundary aşamasında {result['timeout_count']} blok timeout aldı.")
                    st.write("Koleksiyonlar:", f"{index_collection_name}_parents", "/", f"{index_collection_name}_children")
                else:
                    st.success(f"İndeksleme tamamlandı! Toplam **{result['total_chunks']}** chunk indekslendi.")
                    st.markdown("#### Süre Detayları")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("PDF Okuma", f"{result['pdf_extract_sec']}s")
                    col2.metric("Ollama Embed", f"{result['ollama_embed_sec']}s")
                    col3.metric("Qdrant Yazma", f"{result['qdrant_upsert_sec']}s")
                    col4.metric("Toplam", f"{result['total_sec']}s")
                    st.write("Koleksiyon:", index_collection_name)
            except Exception as exc:
                st.error(f"İndeksleme sırasında hata oluştu: {exc}")


def _render_chat_tab(
    connection_error: str | None,
    all_models: List[str],
    filtered_count: int,
    shared_embed_models: List[str],
    collection_name: str,
    openai_api_key: str,
) -> None:
    if connection_error:
        st.error(connection_error)
    _render_qa_model_selector(all_models, filtered_count, key_prefix="chat")
    chat_qa_models_selected = st.session_state.get("chat_qa_models_selected", [])

    with st.container(border=True):
        col_eval, col_embed, col_mode = st.columns([3, 2, 2])

        with col_eval:
            chat_eval_enabled, chat_eval_backend, chat_eval_model_name, chat_local_eval_model_name = _render_eval_settings(all_models, key_prefix="chat")

        with col_embed:
            retrieval_selection = _render_collection_selection(
                shared_embed_models=shared_embed_models,
                collection_name=collection_name,
                key_prefix="chat",
                embed_label="Embedding modeli",
                rag_type_label="Retrieval tipi",
                rag_type_help="Smart parent/child koleksiyonlarını, BM25 ise anahtar kelime aramasını kullanır.",
                rag_type_horizontal=False,
                classic_label="Klasik koleksiyon",
                smart_label="Smart koleksiyon",
                classic_help="İndekslemede kullandığınız koleksiyonu seçin.",
                smart_help="_children/_parents çifti bu base isimden türetilir.",
                classic_caption_prefix="Koleksiyon",
                smart_caption_prefix="Koleksiyonlar",
            )
            chat_embed_model = retrieval_selection.embed_model
            chat_collection_name = retrieval_selection.collection_name
            chat_smart_rag = retrieval_selection.smart_rag
            chat_retrieval_mode = retrieval_selection.retrieval_mode

        with col_mode:
            chat_rag_mode_label = st.radio(
                "Cevaplama modu",
                options=["RAG'li", "RAG'siz", "İkisi birden"],
                horizontal=False,
                key="chat_rag_mode",
            )
            if chat_rag_mode_label != "RAG'siz":
                k_chat = st.number_input(
                    "Context chunk sayısı (k)",
                    min_value=1,
                    max_value=20,
                    value=5,
                    key="chat_k",
                )
            else:
                k_chat = 5
            if chat_rag_mode_label != "RAG'siz" and chat_retrieval_mode == "vector":
                chat_score_threshold = st.slider(
                    "Score threshold",
                    min_value=0.10,
                    max_value=1.0,
                    value=0.55,
                    step=0.05,
                    key="chat_score_threshold",
                    help="Düşük değer daha fazla ama daha zayıf eşleşme döndürür.",
                )
            else:
                chat_score_threshold = 0.55
            chat_thinking_enabled = st.toggle(
                "Thinking modu",
                value=False,
                key="chat_thinking_enabled",
                help=(
                    "Reasoning modellerinde (Qwen3, QwQ vb.) thinking modunu açar. "
                    "Kapalıyken <think> blokları cevaba karışmaz. "
                    "Diğer modeller bu seçeneği zaten görmezden gelir."
                ),
            )

    chat_rag_mode_map = {"RAG'li": "rag", "RAG'siz": "no_rag", "İkisi birden": "both"}
    chat_rag_mode = chat_rag_mode_map[chat_rag_mode_label]

    if "chat_eval_rows" not in st.session_state:
        st.session_state["chat_eval_rows"] = []

    col_q, col_ref = st.columns(2)
    with col_q:
        question = st.text_area(
            "Soru",
            placeholder="Buraya modelden cevap almak istediğin soruyu yaz...",
            height=150,
        )
    with col_ref:
        expected_answer = st.text_area(
            "Beklenen / referans cevap (isteğe bağlı)",
            placeholder="Eval sırasında kıyaslamak için doğru cevabı yazabilirsin.",
            height=150,
        )

    if st.button("Soruyu değerlendir", type="primary", use_container_width=True):
        if not question.strip():
            st.error("Lütfen bir soru gir.")
        elif chat_rag_mode != "no_rag" and not chat_collection_name:
            st.error("Seçili retrieval tipi için geçerli bir koleksiyon bulunamadı.")
        else:
            _run_chat_eval(
                question=question,
                expected_answer=expected_answer,
                rag_mode=chat_rag_mode,
                k=k_chat,
                qa_models_selected=chat_qa_models_selected,
                all_models=all_models,
                eval_enabled=chat_eval_enabled,
                eval_backend=chat_eval_backend,
                eval_model_name=chat_eval_model_name,
                local_eval_model_name=chat_local_eval_model_name,
                openai_api_key=openai_api_key,
                collection_name=chat_collection_name,
                embed_model=chat_embed_model,
                think=chat_thinking_enabled,
                smart_rag=chat_smart_rag,
                score_threshold=float(chat_score_threshold),
                retrieval_mode=chat_retrieval_mode,
            )

    if st.session_state.get("chat_eval_rows"):
        csv_buffer = io.StringIO()
        fieldnames = list(st.session_state["chat_eval_rows"][0].keys())
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        for row in st.session_state["chat_eval_rows"]:
            writer.writerow(row)

        st.download_button(
            label="Manuel chat sonuçlarını CSV olarak indir",
            data=csv_buffer.getvalue().encode("utf-8"),
            file_name="chat_results.csv",
            mime="text/csv",
        )


def _render_voice_tab() -> None:
        st.subheader("Sesli Sentezleme (Sadece TTS)")
        st.write("Bu bölümde yazdığınız veya CSV ile yüklediğiniz metinler doğrudan uzak sunucudaki model ile sese dönüştürülür. LLM veya RAG kullanılmaz.")

        # Uzak sunucuda indirili modelleri çek
        with st.spinner("İndirili TTS modelleri kontrol ediliyor..."):
            downloaded_models = get_downloaded_tts_models()
        
        default_model = "facebook/mms-tts-tur"
        model_options = downloaded_models.copy()
        if default_model not in model_options:
            model_options.insert(0, default_model)
            
        custom_option = "Model Adı Girin"
        if custom_option not in model_options:
            model_options.append(custom_option)

        selected_option = st.selectbox(
            "TTS Modeli Seçin",
            options=model_options,
            help="Sesi sentezlemek için kullanılacak modeli seçin veya yenisini indirmek için 'Yeni Model' seçeneğini kullanın."
        )

        if selected_option == custom_option:
            tts_model_selected = st.text_input("Yeni HuggingFace Model Adını Yazın (örn: facebook/mms-tts-eng):", value="microsoft/speecht5_tts").strip()
        else:
            tts_model_selected = selected_option

        # --- DİNAMİK MİMARİ AYARLARI ---
        st.markdown("#### Modele Özgü Parametreler")
        col_m1, col_m2 = st.columns(2)
        
        speaker_id = None
        voice_preset = None

        model_lower = tts_model_selected.lower()
        
        with col_m1:
            if "speecht5" in model_lower:
                st.info("**SpeechT5 Mimaris:** Bir Speaker ID (0-10000) girerek sesi değiştirebilirsiniz.")
                speaker_id = st.text_input("Speaker ID (Seed)", value="4312", key="speaker_id_input")
            elif "qwen" in model_lower or "fish" in model_lower:
                st.info("**Voice Cloning Mimari:** Belirli bir karakter ID'si veya Stil preset ismi girebilirsiniz.")
                speaker_id = st.text_input("Karakter/Speaker ID", placeholder="Örn: 7", key="cloning_id_input")
            else:
                st.write("Bu model için ek bir parametre gerekmiyor (Standart TTS).")

        with col_m2:
            if "qwen" in model_lower or "fish" in model_lower:
                voice_preset = st.selectbox("Ses Stili / Preset", ["Varsayılan", "Neşeli", "Ciddi", "Fısıltı"], key="voice_preset_sel")
            elif "speecht5" in model_lower:
                # SpeechT5 için yaygın olan bazı presetler veya roller simüle edilebilir
                st.caption("Not: SpeechT5'te ses değişimi için 'Speaker ID' yeterlidir.")

        st.markdown("---")
        st.markdown("### Toplu CSV'den Metin Okuma")
        
        uploaded_voice_csv = st.file_uploader(
            "Metin CSV'si Yükle (Sadece tek sütun ve sadece metinler içermelidir)",
            type=["csv"],
            key="voice_csv_upload"
        )

        if st.button("Toplu CSV'yi İşle ve Sese Çevir", key="voice_csv_btn"):
            if not uploaded_voice_csv:
                st.error("Lütfen bir CSV dosyası yükleyin.")
            else:
                try:
                    stringio = io.StringIO(uploaded_voice_csv.getvalue().decode("utf-8"))
                    reader = csv.reader(stringio)
                    texts_to_read = []
                    for row in reader:
                        if row and row[0].strip():
                            texts_to_read.append(row[0].strip())
                    
                    if not texts_to_read:
                        st.warning("CSV dosyasında geçerli bir metin bulunamadı.")
                    else:
                        st.success(f"Toplam {len(texts_to_read)} adet metin bulundu. Sesli yanıtlar üretiliyor...")
                        for idx, text_content in enumerate(texts_to_read):
                            st.markdown(f"#### Metin {idx + 1}: {text_content}")
                            with st.spinner(f"Metin {tts_model_selected} ile sese çevriliyor..."):
                                wav_bytes, sr, duration_sec = synthesize_speech(
                                    text_content, 
                                    model=tts_model_selected,
                                    speaker_id=speaker_id,
                                    voice_preset=voice_preset
                                )
                            
                            st.write(f"**Ses Uzunluğu:** `{duration_sec:.2f}` saniye | **Model:** {tts_model_selected}")
                            st.audio(wav_bytes, format="audio/wav")
                            st.download_button(
                                f"İndir — Metin {idx+1}",
                                data=wav_bytes,
                                file_name=f"ses_metin_{idx+1}.wav",
                                mime="audio/wav",
                                key=f"voice_bulk_dl_{idx}",
                            )
                            st.markdown("---")
                except Exception as e:
                    st.error(f"CSV işlenirken bir hata oluştu: {e}")

        st.markdown("### Manuel Metin Okuma")

        # --- Section 1: Metin Girişi ---
        transcription = st.text_area(
            "Okunacak metni yazın",
            height=120,
            key="voice_transcription_area",
        )

        if st.button("Sesi Üret (Sentezle)", key="voice_eval_btn"):
            q = transcription.strip()
            if not q:
                st.error("Lütfen bir metin yazın.")
            else:
                st.markdown("---")
                st.markdown("### Sesli Çıktı (TTS)")
                with st.spinner(f"Metin {tts_model_selected} ile sese çevriliyor..."):
                    wav_bytes, sr, duration_sec = synthesize_speech(
                        q, 
                        model=tts_model_selected,
                        speaker_id=speaker_id,
                        voice_preset=voice_preset
                    )
                
                st.write(f"**Ses Uzunluğu:** `{duration_sec:.2f}` saniye | **Parametre:** {f'ID:{speaker_id}' if speaker_id else 'Default'}")
                st.audio(wav_bytes, format="audio/wav")
                st.download_button(
                    "Sesi İndir",
                    data=wav_bytes,
                    file_name="manuel_ses.wav",
                    mime="audio/wav",
                    key="voice_manuel_dl",
                )


def _render_manage_tab() -> None:
        st.subheader("Yönetim")

        col_ollama_mgmt, col_qdrant_mgmt = st.columns(2)

        # ── Ollama Model Yönetimi ──────────────────────────────────────────────
        with col_ollama_mgmt:
            st.markdown("### Ollama Model Yönetimi")

            st.markdown("**Yeni Ollama modeli ekle**")
            col_new_model, col_add_btn = st.columns([4, 1])
            with col_new_model:
                mgmt_new_model_input = st.text_input(
                    "Ollama Model Adı Girin",
                    placeholder="örn: llama3.2:3b",
                    key="mgmt_new_model_input",
                    label_visibility="collapsed",
                )
            with col_add_btn:
                if st.button("Ekle / Pull Et", key="mgmt_add_model_btn"):
                    model_to_add = mgmt_new_model_input.strip()
                    if not model_to_add:
                        st.warning("Model adı boş olamaz.")
                    else:
                        with st.spinner(f"'{model_to_add}' pull ediliyor..."):
                            success, msg = _pull_ollama_model(model_to_add)
                        if success:
                            _list_ollama_models.clear()
                            _list_embedding_models.clear()
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

            # Tüm modelleri (embedding dahil) listele
            host = _get_ollama_base_url()
            all_ollama_names: List[str] = []
            ollama_fetch_err = ""
            if host:
                try:
                    _resp = requests.get(host + "/api/tags", timeout=10)
                    _resp.raise_for_status()
                    all_ollama_names = [
                        item["name"]
                        for item in (_resp.json() or {}).get("models", [])
                        if isinstance(item.get("name"), str)
                    ]
                except Exception as _e:
                    ollama_fetch_err = str(_e)
            else:
                ollama_fetch_err = "OLLAMA_BASE_URL veya OLLAMA_HOST tanımlı değil."

            if ollama_fetch_err:
                st.error(f"Modeller yüklenemedi: {ollama_fetch_err}")
            elif not all_ollama_names:
                st.info("Sunucuda hiç model bulunamadı.")
            else:
                st.divider()
                st.caption(f"Toplam {len(all_ollama_names)} model")
                model_to_delete = st.selectbox(
                    "Silinecek model",
                    options=all_ollama_names,
                    key="mgmt_model_to_delete",
                )
                if "mgmt_model_confirm" not in st.session_state:
                    st.session_state["mgmt_model_confirm"] = False

                if not st.session_state["mgmt_model_confirm"]:
                    if st.button("Modeli Sil", key="mgmt_model_delete_btn", type="primary"):
                        st.session_state["mgmt_model_confirm"] = True
                        st.rerun()
                else:
                    st.warning(f"**'{model_to_delete}'** modelini silmek istediğinizden emin misiniz? Bu işlem geri alınamaz.")
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("Evet, Sil", key="mgmt_model_confirm_yes", type="primary"):
                            success, msg = _delete_ollama_model(model_to_delete)
                            if success:
                                st.success(msg)
                                _list_ollama_models.clear()
                                _list_embedding_models.clear()
                            else:
                                st.error(msg)
                            st.session_state["mgmt_model_confirm"] = False
                            st.rerun()
                    with col_no:
                        if st.button("İptal", key="mgmt_model_confirm_no"):
                            st.session_state["mgmt_model_confirm"] = False
                            st.rerun()

        # ── Qdrant Koleksiyon Yönetimi ─────────────────────────────────────────
        with col_qdrant_mgmt:
            st.markdown("### Qdrant Koleksiyon Yönetimi")

            collections, coll_err = _list_qdrant_collections()

            if coll_err:
                st.error(coll_err)
            elif not collections:
                st.info("Hiç koleksiyon bulunamadı.")
            else:
                st.caption(f"Toplam {len(collections)} koleksiyon")
                coll_to_delete = st.selectbox(
                    "Silinecek koleksiyon",
                    options=collections,
                    key="mgmt_coll_to_delete",
                )
                if "mgmt_coll_confirm" not in st.session_state:
                    st.session_state["mgmt_coll_confirm"] = False

                if not st.session_state["mgmt_coll_confirm"]:
                    if st.button("Koleksiyonu Sil", key="mgmt_coll_delete_btn", type="primary"):
                        st.session_state["mgmt_coll_confirm"] = True
                        st.rerun()
                else:
                    st.warning(f"**'{coll_to_delete}'** koleksiyonunu ve tüm içeriğini silmek istediğinizden emin misiniz? Bu işlem geri alınamaz.")
                    col_yes2, col_no2 = st.columns(2)
                    with col_yes2:
                        if st.button("Evet, Sil", key="mgmt_coll_confirm_yes", type="primary"):
                            success, msg = _delete_qdrant_collection(coll_to_delete)
                            if success:
                                st.success(msg)
                            else:
                                st.error(msg)
                            st.session_state["mgmt_coll_confirm"] = False
                            st.rerun()
                    with col_no2:
                        if st.button("İptal", key="mgmt_coll_confirm_no"):
                            st.session_state["mgmt_coll_confirm"] = False
                            st.rerun()


def _render_analysis_tab() -> None:
        import pandas as pd

        st.subheader("Sonuç Analizi")
        st.caption("Daha önce export edilen CSV dosyasını yükleyerek sonuçları analiz edin.")

        analysis_file = st.file_uploader(
            "Export CSV yükle (noktalı virgül ayraçlı)",
            type=["csv"],
            key="analysis_csv_upload",
        )

        if analysis_file is not None:
            try:
                df = pd.read_csv(analysis_file, sep=";")
            except Exception as _e:
                st.error(f"CSV okunamadı: {_e}")
                df = None

            if df is not None:
                # Kolon varlık kontrolü
                required_cols = {"model", "ai_score", "ai_verdict", "ai_hallucination_risk", "rag_type", "tokens_per_second"}
                missing = required_cols - set(df.columns)
                if missing:
                    st.warning(f"CSV'de eksik kolonlar: {', '.join(sorted(missing))}. Bazı metrikler hesaplanamayabilir.")

                if "ai_score" in df.columns:
                    df["ai_score"] = pd.to_numeric(df["ai_score"], errors="coerce")
                else:
                    df["ai_score"] = pd.Series(float("nan"), index=df.index)

                if "tokens_per_second" in df.columns:
                    df["tokens_per_second"] = pd.to_numeric(df["tokens_per_second"], errors="coerce")
                else:
                    df["tokens_per_second"] = pd.Series(float("nan"), index=df.index)

                st.markdown("---")

                # ── Model Karşılaştırma Tablosu ───────────────────────────────
                st.markdown("### Model Karşılaştırma")

                if "model" in df.columns:
                    rows_model = []
                    for model_name, grp in df.groupby("model"):
                        total = len(grp)
                        row = {"Model": model_name, "Toplam Satır": total}

                        if "ai_score" in df.columns:
                            row["Ort. AI Skoru"] = round(grp["ai_score"].mean(), 2)

                        if "ai_verdict" in df.columns:
                            vc = grp["ai_verdict"].value_counts()
                            row["Correct %"] = round(vc.get("correct", 0) / total * 100, 1)
                            row["Partial %"] = round(vc.get("partial", 0) / total * 100, 1)
                            row["Incorrect %"] = round(vc.get("incorrect", 0) / total * 100, 1)

                        if "ai_hallucination_risk" in df.columns:
                            hc = grp["ai_hallucination_risk"].value_counts()
                            row["Hallucination High %"] = round(hc.get("high", 0) / total * 100, 1)

                        if "tokens_per_second" in df.columns:
                            row["Ort. Token/sn"] = round(grp["tokens_per_second"].mean(), 1)

                        rows_model.append(row)

                    st.dataframe(pd.DataFrame(rows_model).set_index("Model"), use_container_width=True)
                else:
                    st.info("'model' kolonu bulunamadı.")

                st.markdown("---")

                # ── RAG Lift Tablosu ──────────────────────────────────────────
                st.markdown("### RAG Lift (RAG vs RAG'siz)")

                if "model" in df.columns and "rag_type" in df.columns and "ai_score" in df.columns:
                    rag_vals = df["rag_type"].unique().tolist()
                    rag_label = next((v for v in rag_vals if "siz" not in v.lower()), None)
                    no_rag_label = next((v for v in rag_vals if "siz" in v.lower()), None)

                    if rag_label and no_rag_label:
                        lift_rows = []
                        for model_name, grp in df.groupby("model"):
                            rag_grp = grp[grp["rag_type"] == rag_label]
                            no_rag_grp = grp[grp["rag_type"] == no_rag_label]

                            rag_mean = rag_grp["ai_score"].mean()
                            no_rag_mean = no_rag_grp["ai_score"].mean()
                            lift = round(rag_mean - no_rag_mean, 2) if pd.notna(rag_mean) and pd.notna(no_rag_mean) else None

                            # Soru bazında RAG'in hurt ettiği satırlar
                            if "question" in df.columns:
                                merged = rag_grp[["question", "ai_score"]].rename(columns={"ai_score": "rag_score"}).merge(
                                    no_rag_grp[["question", "ai_score"]].rename(columns={"ai_score": "no_rag_score"}),
                                    on="question",
                                    how="inner",
                                )
                                hurt_pct = round(
                                    (merged["no_rag_score"] > merged["rag_score"]).sum() / len(merged) * 100, 1
                                ) if len(merged) > 0 else None
                            else:
                                hurt_pct = None

                            lift_rows.append({
                                "Model": model_name,
                                f"Ort. Skor ({rag_label})": round(rag_mean, 2) if pd.notna(rag_mean) else None,
                                f"Ort. Skor ({no_rag_label})": round(no_rag_mean, 2) if pd.notna(no_rag_mean) else None,
                                "RAG Lift (RAG − RAG'siz)": lift,
                                "RAG'in Hurt Ettiği Soru %": hurt_pct,
                            })

                        st.dataframe(pd.DataFrame(lift_rows).set_index("Model"), use_container_width=True)
                    else:
                        st.info(f"RAG türleri tespit edilemedi. Bulunan değerler: {rag_vals}")
                else:
                    st.info("RAG lift için 'model', 'rag_type' ve 'ai_score' kolonları gerekli.")

                st.markdown("---")

                # ── Soru Bazlı Detay Tablosu ─────────────────────────────────
                st.markdown("### Soru Bazlı Detay")

                filter_cols = st.columns(3)
                filtered_df = df.copy()

                if "model" in df.columns:
                    model_opts = ["(Tümü)"] + sorted(df["model"].dropna().unique().tolist())
                    sel_model = filter_cols[0].selectbox("Model", model_opts, key="analysis_filter_model")
                    if sel_model != "(Tümü)":
                        filtered_df = filtered_df[filtered_df["model"] == sel_model]

                if "ai_verdict" in df.columns:
                    verdict_opts = ["(Tümü)"] + sorted(df["ai_verdict"].dropna().unique().tolist())
                    sel_verdict = filter_cols[1].selectbox("Verdict", verdict_opts, key="analysis_filter_verdict")
                    if sel_verdict != "(Tümü)":
                        filtered_df = filtered_df[filtered_df["ai_verdict"] == sel_verdict]

                if "rag_type" in df.columns:
                    rag_opts = ["(Tümü)"] + sorted(df["rag_type"].dropna().unique().tolist())
                    sel_rag = filter_cols[2].selectbox("RAG Türü", rag_opts, key="analysis_filter_rag")
                    if sel_rag != "(Tümü)":
                        filtered_df = filtered_df[filtered_df["rag_type"] == sel_rag]

                st.caption(f"{len(filtered_df)} satır gösteriliyor (toplam {len(df)})")

                display_cols = [c for c in [
                    "model", "question", "rag_type", "ai_verdict", "ai_score",
                    "ai_hallucination_risk", "tokens_per_second", "eval_duration_seconds",
                    "model_answer", "answer",
                ] if c in filtered_df.columns]
                remaining = [c for c in filtered_df.columns if c not in display_cols]
                st.dataframe(filtered_df[display_cols + remaining].reset_index(drop=True), use_container_width=True)
        else:
            st.info("Analiz etmek istediğiniz export CSV dosyasını yükleyin.")


def main() -> None:
    st.set_page_config(page_title="RAG Değerlendirme Pipeline", layout="wide")

    st.markdown(
        """
        <style>
        div[data-baseweb="popover"] ul {
            background-color: #1e1e1e;
        }
        div[data-baseweb="popover"] li {
            color: #fafafa;
        }
        div[data-baseweb="popover"] li:hover {
            background-color: #333333;
        }
        div[data-baseweb="popover"] li[aria-selected="true"] {
            background-color: #404040;
        }
        div[data-baseweb="select"] > div {
            border-color: #555555;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("RAG + Değerlendirme Pipeline")
    st.write(
        "PDF tabanlı Türkçe RAG sistemi: "
        "PDF'leri Qdrant'a indeksle (Ollama embedding ile), CSV'den soruları değerlendir, "
        "cevapları Ollama ile üret ve değerlendir."
    )

    _render_sidebar_monitor()

    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    qdrant_url = os.environ.get("QDRANT_URL", QDRANT_URL)
    collection_name = os.environ.get("QDRANT_COLLECTION", DEFAULT_COLLECTION_NAME)

    _render_connection_status(qdrant_url)

    ollama_models, connection_error, _filter_elapsed, filtered_count = _list_ollama_models()
    all_models = [] if connection_error else sorted({m for m in ollama_models if m})

    shared_embed_models, shared_embed_err = _list_embedding_models()
    if shared_embed_err or not shared_embed_models:
        _embed_msg = shared_embed_err or "Sunucuda hiç embedding modeli bulunamadı."
        st.error(f"Embedding modelleri yüklenemedi: {_embed_msg} Ollama bağlantısını kontrol et.")

    tab_index, tab_eval, tab_chat, tab_voice, tab_manage, tab_analysis = st.tabs(
        ["PDF İndeksleme", "CSV Değerlendirme", "Manuel Chat Eval", "Sesli Değerlendirme", "Yönetim", "Sonuç Analizi"]
    )

    with tab_index:
        _render_index_tab(
            all_models=all_models,
            shared_embed_models=shared_embed_models,
            collection_name=collection_name,
            qdrant_url=qdrant_url,
        )

    with tab_eval:
        _render_csv_eval_tab(
            connection_error=connection_error,
            all_models=all_models,
            filtered_count=filtered_count,
            shared_embed_models=shared_embed_models,
            collection_name=collection_name,
        )

    with tab_chat:
        _render_chat_tab(
            connection_error=connection_error,
            all_models=all_models,
            filtered_count=filtered_count,
            shared_embed_models=shared_embed_models,
            collection_name=collection_name,
            openai_api_key=openai_api_key,
        )

    with tab_voice:
        _render_voice_tab()

    with tab_manage:
        _render_manage_tab()

    with tab_analysis:
        _render_analysis_tab()


if __name__ == "__main__":
    main()
