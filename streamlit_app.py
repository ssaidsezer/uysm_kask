from __future__ import annotations

import csv
import io
import os
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
    index_pdfs,
    retrieve_context,
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
    write_results_to_csv,
)
from voice_utils import synthesize_speech, get_downloaded_tts_models


def _is_embedding_model(host: str, model_name: str) -> bool:
    """
    /api/show ile modelin embedding modeli olup olmadığını kontrol eder.
    Embedding modelleri: template alanı boştur ve/veya model_info içinde
    architecture olarak 'bert' vb. embedding mimarileri bulunur.
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

    # Model adında "embed" geçiyorsa doğrudan embedding say
    if "embed" in model_name.lower():
        return True

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
    host = os.environ.get("OLLAMA_HOST", "")
    if not host:
        return [], "OLLAMA_HOST ortam değişkeni tanımlı değil. Lütfen .env dosyasına uzak sunucu adresini ekleyin (örn: OLLAMA_HOST=192.168.1.151:11434).", 0.0, 0

    if not host.startswith("http"):
        host = f"http://{host}"
    url = host.rstrip("/") + "/api/tags"

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
    host = os.environ.get("OLLAMA_HOST", "")
    if not host:
        return [], "OLLAMA_HOST ortam değişkeni tanımlı değil."

    if not host.startswith("http"):
        host = f"http://{host}"
    url = host.rstrip("/") + "/api/tags"

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
    host = os.environ.get("OLLAMA_HOST", "")
    if not host:
        return False, "OLLAMA_HOST ortam değişkeni tanımlı değil."
    if not host.startswith("http"):
        host = f"http://{host}"
    url = host.rstrip("/") + "/api/pull"
    try:
        resp = requests.post(url, json={"name": model_name, "stream": False}, timeout=300)
        resp.raise_for_status()
        return True, f"'{model_name}' başarıyla pull edildi."
    except Exception as e:
        return False, f"Pull sırasında hata: {e}"


def _run_chat_eval(
    question: str,
    expected_answer: str,
    rag_mode: str,
    k: int,
    qa_models_selected: List[str],
    all_models: List[str],
    eval_backend: str,
    eval_model_name: str,
    local_eval_model_name: str | None,
    openai_api_key: str,
    collection_name: str,
) -> List[dict]:
    """Run RAG (and optionally no-RAG) QA + evaluation for given models.

    Displays results via Streamlit and appends to session state.
    Returns list of result dicts containing 'model_answer' for TTS.
    """
    if eval_backend == "OpenAI":
        if not openai_api_key and not os.environ.get("OPENAI_API_KEY"):
            st.error(
                "OpenAI değerlendirme motoru seçili. OpenAI API key gerekli "
                "(sidebar'dan girin veya ortam değişkeni ayarlayın)."
            )
            return []
        openai_client = get_openai_client(api_key=openai_api_key or None)
    else:
        openai_client = None

    context = retrieve_context(
        question=question,
        collection_name=collection_name,
        k=int(k),
    )

    run_timestamp = datetime.utcnow().isoformat()
    selected_models = qa_models_selected or all_models
    answers: List[dict] = []

    for qa_model_name in selected_models:
        warmup_model(model=qa_model_name)
        st.markdown(f"**Model: {qa_model_name}**")
        cols = st.columns(2) if rag_mode == "both" else [st.container()]

        # --- RAG'li cevap ---
        if rag_mode in ("rag", "both"):
            try:
                with st.spinner(
                    f"{qa_model_name} için RAG'li cevap üretiliyor ve değerlendiriliyor..."
                ):
                    rag_result = generate_rag_answer_ollama(
                        question=question,
                        context=context,
                        model=qa_model_name,
                    )
                    rag_record = {
                        "model": f"{qa_model_name} (RAG)" if rag_mode == "both" else qa_model_name,
                        "question_index": 0,
                        "question": question,
                        "observation_idea": expected_answer or "",
                        "model_answer": rag_result.get("answer", ""),
                        "response_time_seconds": rag_result.get("response_time_seconds", 0.0),
                    }
                    rag_eval = evaluate_answer_any(
                        record=rag_record,
                        eval_model=eval_model_name,
                        client=openai_client,
                        backend="openai" if eval_backend == "OpenAI" else "ollama",
                        local_model=local_eval_model_name,
                    )
            except Exception as exc:
                st.error(
                    f"{qa_model_name} için RAG'li çağrıda hata oluştu ve model atlandı: {exc}"
                )
                continue

            with cols[0]:
                st.markdown("**RAG'li cevap (Ollama)**")
                st.write(rag_record["model_answer"])
                st.markdown("**RAG'li cevap eval**")
                st.json(rag_eval)

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
                    "ai_score": rag_eval.get("ai_score", ""),
                    "ai_verdict": rag_eval.get("ai_verdict", ""),
                    "ai_hallucination_risk": rag_eval.get("ai_hallucination_risk", ""),
                }
            )
            answers.append(
                {"model": qa_model_name, "mode": "RAG", "answer": rag_record["model_answer"]}
            )

        # --- RAG'siz cevap ---
        if rag_mode in ("no_rag", "both"):
            try:
                with st.spinner(
                    f"{qa_model_name} için RAG'siz cevap üretiliyor ve değerlendiriliyor..."
                ):
                    no_rag_result = generate_no_rag_answer_ollama(
                        question=question,
                        model=qa_model_name,
                    )
                    no_rag_record = {
                        "model": f"{qa_model_name} (RAG'siz)" if rag_mode == "both" else qa_model_name,
                        "question_index": 0,
                        "question": question,
                        "observation_idea": expected_answer or "",
                        "model_answer": no_rag_result.get("answer", ""),
                        "response_time_seconds": no_rag_result.get("response_time_seconds", 0.0),
                    }
                    no_rag_eval = evaluate_answer_any(
                        record=no_rag_record,
                        eval_model=eval_model_name,
                        client=openai_client,
                        backend="openai" if eval_backend == "OpenAI" else "ollama",
                        local_model=local_eval_model_name,
                    )
            except Exception as exc:
                st.error(
                    f"{qa_model_name} için RAG'siz çağrıda hata oluştu ve bu mod atlandı: {exc}"
                )
                continue

            with cols[1] if rag_mode == "both" else cols[0]:
                st.markdown("**RAG'siz cevap (Ollama)**")
                st.write(no_rag_record["model_answer"])
                st.markdown("**RAG'siz cevap eval**")
                st.json(no_rag_eval)

            st.session_state.setdefault("chat_eval_rows", []).append(
                {
                    "timestamp": run_timestamp,
                    "model": qa_model_name,
                    "mode": "NO_RAG",
                    "question": question,
                    "expected_answer": (expected_answer or "").strip(),
                    "model_answer": no_rag_record["model_answer"],
                    "response_time_seconds": no_rag_record["response_time_seconds"],
                    "tokens_per_second": no_rag_result.get("tokens_per_second") or "",
                    "ai_score": no_rag_eval.get("ai_score", ""),
                    "ai_verdict": no_rag_eval.get("ai_verdict", ""),
                    "ai_hallucination_risk": no_rag_eval.get("ai_hallucination_risk", ""),
                }
            )
            answers.append(
                {"model": qa_model_name, "mode": "NO_RAG", "answer": no_rag_record["model_answer"]}
            )

    return answers


def _render_qa_model_selector(all_models: List[str], filtered_count: int, key_prefix: str) -> List[str]:
    """QA model seçim UI'ı render eder, seçili modelleri döndürür."""
    search_key = f"{key_prefix}_qa_model_search"
    expander_key = f"{key_prefix}_qa_expander_open"
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

    if expander_key not in st.session_state:
        st.session_state[expander_key] = False

    qa_models_selected: List[str] = []
    with st.expander(expander_label, expanded=st.session_state[expander_key]):
        st.session_state[expander_key] = True
        st.text_input("Model ara", placeholder="Model adında ara...", key=search_key)
        col_sel, col_desel, col_empty = st.columns([1, 1, 8])
        with col_sel:
            st.button("Hepsini seç", key=f"{key_prefix}_qa_select_all", on_click=_select_all)
        with col_desel:
            st.button("Hepsini kaldır", key=f"{key_prefix}_qa_deselect_all", on_click=_deselect_all)
        grid_cols = st.columns(3)
        for i, model_name in enumerate(filtered_models):
            with grid_cols[i % 3]:
                if st.checkbox(
                    model_name,
                    value=st.session_state.get(f"{key_prefix}_qa_model_select_{model_name}", False),
                    key=f"{key_prefix}_qa_model_select_{model_name}",
                    help="Bu modeli RAG değerlendirmesine dahil et.",
                ):
                    qa_models_selected.append(model_name)

    # Yeni Ollama modeli ekleme / pull
    st.markdown("**Yeni Ollama modeli ekle**")
    col_new_model, col_add_btn = st.columns([4, 1])
    with col_new_model:
        new_model_input = st.text_input(
            "Ollama Model Adı Girin",
            placeholder="örn: llama3.2:3b",
            key=f"{key_prefix}_new_model_input",
            label_visibility="collapsed",
        )
    with col_add_btn:
        if st.button("Ekle / Pull Et", key=f"{key_prefix}_add_model_btn"):
            model_to_add = new_model_input.strip()
            if not model_to_add:
                st.warning("Model adı boş olamaz.")
            elif model_to_add in combined_models:
                st.info(f"'{model_to_add}' zaten listede mevcut.")
            else:
                with st.spinner(f"'{model_to_add}' pull ediliyor..."):
                    success, msg = _pull_ollama_model(model_to_add)
                if success:
                    st.session_state[custom_models_key].append(model_to_add)
                    st.session_state[f"{key_prefix}_qa_model_select_{model_to_add}"] = True
                    _list_ollama_models.clear()
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    return qa_models_selected


def _render_eval_settings(all_models: List[str], key_prefix: str):
    """Değerlendirme motoru ayarlarını render eder. (eval_backend, eval_model_name, local_eval_model_name) döndürür."""
    col_backend, col_model = st.columns(2)
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
    return eval_backend, eval_model_name, local_eval_model_name


def main() -> None:
    st.set_page_config(page_title="RAG Değerlendirme Pipeline", layout="wide")

    # Selectbox dropdown stilini düzelt (koyu temada okunabilirlik)
    st.markdown(
        """
        <style>
        /* Dropdown listesi arka planı ve metin rengi */
        div[data-baseweb="popover"] ul {
            background-color: #1e1e1e;
        }
        div[data-baseweb="popover"] li {
            color: #fafafa;
        }
        div[data-baseweb="popover"] li:hover {
            background-color: #333333;
        }
        /* Seçili öğe vurgusu */
        div[data-baseweb="popover"] li[aria-selected="true"] {
            background-color: #404040;
        }
        /* Selectbox border rengini yumuşat */
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

    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    qdrant_url = os.environ.get("QDRANT_URL", QDRANT_URL)
    collection_name = os.environ.get("QDRANT_COLLECTION", DEFAULT_COLLECTION_NAME)

    conn_cols = st.columns(2)
    with conn_cols[0]:
        try:
            resp = requests.get(f"{qdrant_url}/collections", timeout=5)
            resp.raise_for_status()
            st.success(f"Qdrant bağlı ({qdrant_url})")
        except Exception:
            st.error(f"Qdrant erişilemiyor ({qdrant_url})")

    ollama_base = os.environ.get("OLLAMA_BASE_URL", "")
    if ollama_base:
        with conn_cols[1]:
            try:
                resp = requests.get(f"{ollama_base.rstrip('/')}/api/tags", timeout=5)
                resp.raise_for_status()
                st.success(f"Ollama bağlı ({ollama_base})")
            except Exception:
                st.error(f"Ollama erişilemiyor ({ollama_base})")

    # Ollama model listesi — cache'li, iki tab'da paylaşılır
    ollama_models, connection_error, _filter_elapsed, filtered_count = _list_ollama_models()
    if connection_error:
        all_models: List[str] = []
    else:
        all_models = sorted({m for m in ollama_models if m})

    tab_index, tab_eval, tab_chat, tab_voice = st.tabs(
        ["📚 PDF İndeksleme", "📊 CSV Değerlendirme", "💬 Manuel Chat Eval", "🎙️ Sesli Değerlendirme"]
    )

    # =========================================================================
    # TAB 1: PDF İndeksleme (Qdrant + Ollama Embedding)
    # =========================================================================
    with tab_index:
        st.subheader("PDF'leri Qdrant'a indeksle (Ollama Embedding)")

        uploaded_pdfs = st.file_uploader(
            "PDF yükle",
            type=["pdf"],
            accept_multiple_files=True,
        )

        chunk_size = int(os.environ.get("CHUNK_SIZE", "1000"))
        chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", "200"))

        with st.spinner("Embedding modelleri yükleniyor..."):
            embed_models, embed_err = _list_embedding_models()

        default_embed = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        if embed_err:
            st.warning(f"Embedding modelleri listelenemedi: {embed_err}")
            embed_model_name = default_embed
        elif not embed_models:
            st.warning("Sunucuda embedding modeli bulunamadı. Varsayılan kullanılıyor.")
            embed_model_name = default_embed
        else:
            default_index = embed_models.index(default_embed) if default_embed in embed_models else 0
            embed_model_name = st.selectbox(
                "Embedding modeli",
                options=embed_models,
                index=default_index,
                help="PDF'leri indekslemek için kullanılacak Ollama embedding modeli.",
            )

        st.info(f"Embedding modeli: **{embed_model_name}** (Ollama) | Koleksiyon: **{collection_name}** | Qdrant: **{qdrant_url}**")

        if st.button("İndeksi oluştur / güncelle"):
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

                    phase_labels = {
                        "pdf_extract": "📄 PDF'ler okunuyor",
                        "ollama_embed": "🧠 Ollama embedding hesaplanıyor",
                        "qdrant_upsert": "💾 Qdrant'a yazılıyor",
                    }
                    # Ağırlıklar: embedding en uzun süren, ona en çok pay ver
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
                        overall = phase_starts.get(phase, 0) + phase_weights.get(phase, 0) * pct_in_phase
                        overall = min(overall, 1.0)
                        progress_bar.progress(overall, text=f"{label}  ({current}/{total})  ⏱ {elapsed_sec:.1f}s")
                        status_text.caption(f"{label}: {current}/{total} — {elapsed_sec:.1f} saniye")

                    result = index_pdfs(
                        [str(p) for p in pdf_paths],
                        collection_name=collection_name,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        qdrant_url=qdrant_url,
                        embed_model=embed_model_name,
                        progress_callback=on_progress,
                    )

                    progress_bar.progress(1.0, text="✅ Tamamlandı!")
                    status_text.empty()

                    st.success(f"İndeksleme tamamlandı! Toplam **{result['total_chunks']}** chunk indekslendi.")

                    # Zamanlama tablosu
                    st.markdown("#### ⏱ Süre Detayları")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("📄 PDF Okuma", f"{result['pdf_extract_sec']}s")
                    col2.metric("🧠 Ollama Embed", f"{result['ollama_embed_sec']}s")
                    col3.metric("💾 Qdrant Yazma", f"{result['qdrant_upsert_sec']}s")
                    col4.metric("⏱ Toplam", f"{result['total_sec']}s")

                    st.write("Koleksiyon:", collection_name)
                except Exception as exc:
                    st.error(f"İndeksleme sırasında hata oluştu: {exc}")

    # =========================================================================
    # TAB 2: CSV Değerlendirme
    # =========================================================================
    with tab_eval:
        st.subheader("CSV'den soruları değerlendir")

        st.markdown("**Değerlendirilecek QA modelleri**")
        if connection_error:
            st.error(connection_error)
        qa_models_selected = _render_qa_model_selector(all_models, filtered_count, key_prefix="csv")

        st.markdown("---")
        eval_backend, eval_model_name, local_eval_model_name = _render_eval_settings(all_models, key_prefix="csv")
        st.markdown("---")

        uploaded_csv = st.file_uploader(
            "CSV yükle (Questions, Answers kolonları içermeli)",
            type=["csv"],
        )

        sample_csv_path = WORKSPACE_DIR / "sample_rag_input.csv"
        use_sample = False
        if sample_csv_path.exists():
            use_sample = st.checkbox(
                "Varsayılan örnek CSV'yi kullan (sample_rag_input.csv)",
                value=not uploaded_csv,
            )

        col_k, col_rag_mode = st.columns(2)
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

        if st.button("Pipeline'ı çalıştır"):
            csv_path: Path

            if uploaded_csv is not None:
                tmp_dir = _ensure_tmp_dir()
                csv_path = tmp_dir / "uploaded_input.csv"
                with csv_path.open("wb") as f:
                    f.write(uploaded_csv.getbuffer())
            elif use_sample and sample_csv_path.exists():
                csv_path = sample_csv_path
            else:
                st.error("CSV seçilmedi.")
                return

            if eval_backend == "OpenAI":
                if not openai_api_key and not os.environ.get("OPENAI_API_KEY"):
                    st.error(
                        "OpenAI değerlendirme motoru seçili. OpenAI API key gerekli."
                    )
                    return
                client = get_openai_client(api_key=openai_api_key or None)
            else:
                client = None

            models_to_run = qa_models_selected if qa_models_selected else ([QA_OLLAMA_MODEL] if not all_models else [all_models[0]])
            rows = []
            pipeline_error = False
            for qa_model in models_to_run:
                try:
                    with st.spinner(f"Pipeline çalışıyor: {qa_model} ({rag_mode_label})..."):
                        model_rows = run_full_pipeline(
                            csv_path=str(csv_path),
                            collection_name=collection_name,
                            qdrant_url=qdrant_url,
                            eval_model=eval_model_name,
                            k=int(k),
                            openai_client=client,
                            eval_backend="openai" if eval_backend == "OpenAI" else "ollama",
                            eval_local_model=local_eval_model_name,
                            qa_model=qa_model,
                            rag_mode=rag_mode,
                        )
                        rows.extend(model_rows)
                except Exception as exc:
                    st.error(f"{qa_model} için pipeline çalışırken hata oluştu: {exc}")
                    pipeline_error = True
            if pipeline_error and not rows:
                return

            if not rows:
                st.warning("Hiç satır üretilmedi.")
                return

            st.success(f"Pipeline tamamlandı. Toplam {len(rows)} satır üretildi.")
            st.dataframe(rows)

            output_csv = io.StringIO()
            _ = write_results_to_csv(rows, output_path=output_csv)

            csv_bytes = output_csv.getvalue().encode("utf-8")
            st.download_button(
                label="Sonuç CSV'yi indir",
                data=csv_bytes,
                file_name="output.csv",
                mime="text/csv",
            )

    # =========================================================================
    # TAB 3: Manuel Chat Eval
    # =========================================================================
    with tab_chat:
        st.subheader("Manuel soru sor ve RAG vs RAG'siz karşılaştır")

        st.markdown("**Değerlendirilecek QA modelleri**")
        if connection_error:
            st.error(connection_error)
        chat_qa_models_selected = _render_qa_model_selector(all_models, filtered_count, key_prefix="chat")

        st.markdown("---")
        chat_eval_backend, chat_eval_model_name, chat_local_eval_model_name = _render_eval_settings(all_models, key_prefix="chat")
        st.markdown("---")

        if "chat_eval_rows" not in st.session_state:
            st.session_state["chat_eval_rows"] = []

        col_q, col_ref = st.columns(2)
        with col_q:
            question = st.text_area(
                "Soru",
                placeholder="Buraya modelden cevap almak istediğin soruyu yaz...",
                height=120,
            )
        with col_ref:
            expected_answer = st.text_area(
                "İsteğe bağlı: Beklenen / referans cevap",
                placeholder="Eval sırasında kıyaslamak için isteğe bağlı olarak doğru cevabı yazabilirsin.",
                height=120,
            )

        chat_rag_mode_label = st.radio(
            "Cevaplama modu",
            options=["RAG'li", "RAG'siz", "İkisi birden"],
            horizontal=True,
            key="chat_rag_mode",
        )
        chat_rag_mode_map = {"RAG'li": "rag", "RAG'siz": "no_rag", "İkisi birden": "both"}
        chat_rag_mode = chat_rag_mode_map[chat_rag_mode_label]

        k_chat = st.number_input(
            "RAG için alınacak context chunk sayısı (k)",
            min_value=1,
            max_value=20,
            value=5,
        )

        if st.button("Soruyu değerlendir"):
            if not question.strip():
                st.error("Lütfen bir soru gir.")
                return

            _run_chat_eval(
                question=question,
                expected_answer=expected_answer,
                rag_mode=chat_rag_mode,
                k=k_chat,
                qa_models_selected=chat_qa_models_selected,
                all_models=all_models,
                eval_backend=chat_eval_backend,
                eval_model_name=chat_eval_model_name,
                local_eval_model_name=chat_local_eval_model_name,
                openai_api_key=openai_api_key,
                collection_name=collection_name,
            )

        # Manuel chat logunu CSV olarak indirme
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

    # ── 4th Tab: Sesli Değerlendirme ( Voice ) ──────────────────────────
    with tab_voice:
        st.subheader("Sesli Sentezleme (Sadece TTS)")
        st.write("Bu bölümde yazdığınız veya CSV ile yüklediğiniz metinler doğrudan uzak sunucudaki model ile sese dönüştürülür. LLM veya RAG kullanılmaz.")

        # Uzak sunucuda indirili modelleri çek
        with st.spinner("İndirili TTS modelleri kontrol ediliyor..."):
            downloaded_models = get_downloaded_tts_models()
        
        default_model = "facebook/mms-tts-tur"
        model_options = downloaded_models.copy()
        if default_model not in model_options:
            model_options.insert(0, default_model)
            
        custom_option = "📝 Model Adı Girin"
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
        st.markdown("#### ⚙️ Modele Özgü Parametreler")
        col_m1, col_m2 = st.columns(2)
        
        speaker_id = None
        voice_preset = None

        model_lower = tts_model_selected.lower()
        
        with col_m1:
            if "speecht5" in model_lower:
                st.info("💡 **SpeechT5 Mimaris:** Bir Speaker ID (0-10000) girerek sesi değiştirebilirsiniz.")
                speaker_id = st.text_input("Speaker ID (Seed)", value="4312", key="speaker_id_input")
            elif "qwen" in model_lower or "fish" in model_lower:
                st.info("💡 **Voice Cloning Mimari:** Belirli bir karakter ID'si veya Stil preset ismi girebilirsiniz.")
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
        st.markdown("### 📄 Toplu CSV'den Metin Okuma")
        
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
                            st.markdown(f"#### 🟡 Metin {idx + 1}: {text_content}")
                            with st.spinner(f"Metin {tts_model_selected} ile sese çevriliyor..."):
                                wav_bytes, sr, duration_sec = synthesize_speech(
                                    text_content, 
                                    model=tts_model_selected,
                                    speaker_id=speaker_id,
                                    voice_preset=voice_preset
                                )
                            
                            st.write(f"⏱️ **Ses Uzunluğu:** `{duration_sec:.2f}` saniye | **Model:** {tts_model_selected}")
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

        st.markdown("### ✍️ Manuel Metin Okuma")

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
                st.markdown("### 🔊 Sesli Çıktı (TTS)")
                with st.spinner(f"Metin {tts_model_selected} ile sese çevriliyor..."):
                    wav_bytes, sr, duration_sec = synthesize_speech(
                        q, 
                        model=tts_model_selected,
                        speaker_id=speaker_id,
                        voice_preset=voice_preset
                    )
                
                st.write(f"⏱️ **Ses Uzunluğu:** `{duration_sec:.2f}` saniye | **Parametre:** {f'ID:{speaker_id}' if speaker_id else 'Default'}")
                st.audio(wav_bytes, format="audio/wav")
                st.download_button(
                    "Sesi İndir",
                    data=wav_bytes,
                    file_name="manuel_ses.wav",
                    mime="audio/wav",
                    key="voice_manuel_dl",
                )


if __name__ == "__main__":
    main()
