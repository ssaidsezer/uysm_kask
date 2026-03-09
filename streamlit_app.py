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
    evaluate_answer,
    evaluate_answer_any,
    generate_rag_answer_ollama,
    generate_no_rag_answer_ollama,
    get_openai_client,
    run_full_pipeline,
    write_results_to_csv,
)
from voice_utils import synthesize_speech


def _list_ollama_models() -> tuple[List[str], str]:
    """
    List models from the remote Ollama HTTP API (/api/tags).
    OLLAMA_HOST must be set in .env — no localhost fallback.
    """
    host = os.environ.get("OLLAMA_HOST", "")
    if not host:
        return [], "OLLAMA_HOST ortam değişkeni tanımlı değil. Lütfen .env dosyasına uzak sunucu adresini ekleyin (örn: OLLAMA_HOST=192.168.1.151:11434)."

    if not host.startswith("http"):
        host = f"http://{host}"
    url = host.rstrip("/") + "/api/tags"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json() or {}
    except Exception:
        return [], f"Uzak Ollama sunucusuna ({host}) bağlanılamadı. Lütfen sunucunun açık olduğundan ve ağ/güvenlik duvarı ayarlarının yapıldığından emin olun."

    models: List[str] = []
    for item in data.get("models", []):
        name = item.get("name")
        if isinstance(name, str):
            models.append(name)
    return models, ""


def _default_pdf_paths() -> List[Path]:
    candidates = [
        WORKSPACE_DIR / "askeri_egitim_kitabi.pdf",
        WORKSPACE_DIR / "taktik_muharebe_yarali_bakimi_el_kitabi.pdf",
    ]
    return [p for p in candidates if p.exists()]


def _ensure_tmp_dir() -> Path:
    tmp_dir = WORKSPACE_DIR / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    return tmp_dir


def _run_chat_eval(
    question: str,
    expected_answer: str,
    compare_no_rag: bool,
    k: int,
    qa_models_selected: List[str],
    all_models: List[str],
    eval_backend: str,
    eval_model_name: str,
    local_eval_model_name: str | None,
    openai_api_key: str,
    chroma_dir: str,
    collection_name: str,
) -> List[dict]:
    """Run RAG (and optionally no-RAG) QA + evaluation for given models.

    Displays results via Streamlit and appends to session state.
    Returns list of result dicts containing 'model_answer' for TTS.
    """
    from rag_index import retrieve_context

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

    client = get_chroma_client(persist_directory=chroma_dir)
    collection = get_or_create_collection(client, name=collection_name)
    embedding_model = load_embedding_model()

    context = retrieve_context(
        question=question,
        collection=collection,
        embedding_model=embedding_model,
        k=int(k),
    )

    run_timestamp = datetime.utcnow().isoformat()
    selected_models = qa_models_selected or all_models
    answers: List[dict] = []

    for qa_model_name in selected_models:
        st.markdown(f"**Model: {qa_model_name}**")
        cols = st.columns(2) if compare_no_rag else [st.container()]

        # --- RAG'li cevap ---
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
                    "model": f"{qa_model_name} (RAG)",
                    "question_index": 0,
                    "question": question,
                    "observation_idea": expected_answer or "",
                    "model_answer": rag_result.get("answer", ""),
                    "response_time_seconds": rag_result.get(
                        "response_time_seconds", 0.0
                    ),
                }

                rag_eval = evaluate_answer_any(
                    record=rag_record,
                    eval_model=eval_model_name,
                    client=openai_client,
                    backend="openai"
                    if eval_backend == "OpenAI"
                    else "ollama",
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
            st.markdown("**RAG'li cevap eval (OpenAI)**")
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
                "ai_hallucination_risk": rag_eval.get(
                    "ai_hallucination_risk", ""
                ),
            }
        )

        answers.append(
            {"model": qa_model_name, "mode": "RAG", "answer": rag_record["model_answer"]}
        )

        # --- RAG'siz cevap ---
        if compare_no_rag:
            try:
                with st.spinner(
                    f"{qa_model_name} için RAG'siz cevap üretiliyor ve değerlendiriliyor..."
                ):
                    no_rag_result = generate_no_rag_answer_ollama(
                        question=question,
                        model=qa_model_name,
                    )

                    no_rag_record = {
                        "model": f"{qa_model_name} (RAG'siz)",
                        "question_index": 0,
                        "question": question,
                        "observation_idea": expected_answer or "",
                        "model_answer": no_rag_result.get("answer", ""),
                        "response_time_seconds": no_rag_result.get(
                            "response_time_seconds", 0.0
                        ),
                    }

                    no_rag_eval = evaluate_answer_any(
                        record=no_rag_record,
                        eval_model=eval_model_name,
                        client=openai_client,
                        backend="openai"
                        if eval_backend == "OpenAI"
                        else "ollama",
                        local_model=local_eval_model_name,
                    )
            except Exception as exc:
                st.error(
                    f"{qa_model_name} için RAG'siz çağrıda hata oluştu ve bu mod atlandı: {exc}"
                )
                continue

            with cols[1]:
                st.markdown("**RAG'siz cevap (Ollama)**")
                st.write(no_rag_record["model_answer"])
                st.markdown("**RAG'siz cevap eval (OpenAI)**")
                st.json(no_rag_eval)

            st.session_state.setdefault("chat_eval_rows", []).append(
                {
                    "timestamp": run_timestamp,
                    "model": qa_model_name,
                    "mode": "NO_RAG",
                    "question": question,
                    "expected_answer": (expected_answer or "").strip(),
                    "model_answer": no_rag_record["model_answer"],
                    "response_time_seconds": no_rag_record[
                        "response_time_seconds"
                    ],
                    "tokens_per_second": no_rag_result.get(
                        "tokens_per_second"
                    )
                    or "",
                    "ai_score": no_rag_eval.get("ai_score", ""),
                    "ai_verdict": no_rag_eval.get("ai_verdict", ""),
                    "ai_hallucination_risk": no_rag_eval.get(
                        "ai_hallucination_risk", ""
                    ),
                }
            )

            answers.append(
                {"model": qa_model_name, "mode": "NO_RAG", "answer": no_rag_record["model_answer"]}
            )

    return answers


def main() -> None:
    st.set_page_config(page_title="RAG Değerlendirme Pipeline", layout="wide")

    st.title("RAG + Değerlendirme Pipeline")
    st.write(
        "PDF tabanlı Türkçe RAG sistemi: "
        "PDF'leri Qdrant'a indeksle (Ollama embedding ile), CSV'den soruları değerlendir, "
        "cevapları Ollama ile üret ve değerlendir."
    )

    with st.sidebar:
        st.header("Ayarlar")

        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        qdrant_url = os.environ.get("QDRANT_URL", QDRANT_URL)
        collection_name = os.environ.get("QDRANT_COLLECTION", DEFAULT_COLLECTION_NAME)

        # Bağlantı durumu göstergesi
        st.markdown("---")
        st.markdown("**Bağlantı Durumu**")
        # Qdrant
        try:
            resp = requests.get(f"{qdrant_url}/collections", timeout=5)
            resp.raise_for_status()
            st.success(f"✅ Qdrant bağlı ({qdrant_url})")
        except Exception:
            st.error(f"❌ Qdrant erişilemiyor ({qdrant_url})")

        # Ollama
        ollama_base = os.environ.get("OLLAMA_BASE_URL", "")
        if ollama_base:
            try:
                resp = requests.get(f"{ollama_base.rstrip('/')}/api/tags", timeout=5)
                resp.raise_for_status()
                st.success(f"✅ Ollama bağlı ({ollama_base})")
            except Exception:
                st.error(f"❌ Ollama erişilemiyor ({ollama_base})")
        st.markdown("---")

        # Modelleri yalnızca uzak Ollama sunucusundan listele
        with st.spinner("Ollama modelleri listeleniyor..."):
            ollama_models, connection_error = _list_ollama_models()

        if connection_error:
            st.error(connection_error)
            all_models = []
        else:
            all_models: List[str] = sorted({m for m in ollama_models if m})
            if not all_models:
                st.warning("Ollama sunucusuna bağlanıldı ancak herhangi bir model bulunamadı.")

        # Değerlendirilecek QA modelleri
        st.markdown("**Değerlendirilecek QA modelleri**")

        qa_models_selected: List[str] = []
        with st.expander("Modelleri göster", expanded=False):
            qa_model_search = st.text_input(
                "Model ara",
                value=st.session_state.get("qa_model_search", ""),
                placeholder="Model adında ara...",
                key="qa_model_search",
            )

            search_value = st.session_state.get("qa_model_search", "")
            if search_value:
                filtered_models = [
                    m for m in all_models if search_value.lower() in m.lower()
                ]
            else:
                filtered_models = all_models

            col_sel_all, col_desel_all = st.columns(2)
            with col_sel_all:
                if st.button("Hepsini seç", key="qa_select_all"):
                    for m in filtered_models:
                        st.session_state[f"qa_model_select_{m}"] = True
            with col_desel_all:
                if st.button("Hepsini kaldır", key="qa_deselect_all"):
                    for m in filtered_models:
                        st.session_state[f"qa_model_select_{m}"] = False

            for model_name in filtered_models:
                if st.checkbox(
                    model_name,
                    value=st.session_state.get(f"qa_model_select_{model_name}", True),
                    key=f"qa_model_select_{model_name}",
                    help="Bu modeli RAG değerlendirmesine dahil et.",
                ):
                    qa_models_selected.append(model_name)

        eval_backend = st.selectbox(
            "Değerlendirme motoru",
            options=["OpenAI", "Yerel (Ollama)"],
            index=0,
            help="Cevapları OpenAI ile mi yoksa yerel bir Ollama modeliyle mi değerlendireceğini seç.",
        )

        local_eval_model_name: str | None = None
        if eval_backend == "OpenAI":
            eval_model_name = st.text_input(
                "OpenAI değerlendirme modeli",
                value=EVAL_MODEL_NAME,
                help="OpenAI değerlendirme motoru seçiliyse kullanılacak model.",
            )
        else:
            eval_model_name = EVAL_MODEL_NAME
            local_eval_model_name = st.selectbox(
                "Yerel değerlendirme modeli (Ollama)",
                options=all_models if all_models else ["Bağlantı hatası/Model Yok"],
                index=0,
                help="Eval için kullanılacak yerel Ollama modelini seç.",
            )

    tab_index, tab_eval, tab_chat, tab_voice = st.tabs(
        ["📚 PDF İndeksleme", "📊 CSV Değerlendirme", "💬 Manuel Chat Eval", "🎙️ Sesli Değerlendirme"]
    )

    # =========================================================================
    # TAB 1: PDF İndeksleme (Qdrant + Ollama Embedding)
    # =========================================================================
    with tab_index:
        st.subheader("PDF'leri Qdrant'a indeksle (Ollama Embedding)")

        default_pdfs = _default_pdf_paths()
        use_default = False
        if default_pdfs:
            use_default = st.checkbox(
                "Varsayılan PDF'leri kullan",
                value=True,
                help=", ".join(str(p.name) for p in default_pdfs),
            )

        uploaded_pdfs = st.file_uploader(
            "Ek PDF yükle (isteğe bağlı)",
            type=["pdf"],
            accept_multiple_files=True,
        )

        chunk_size = int(os.environ.get("CHUNK_SIZE", "1000"))
        chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", "200"))

        embed_model_name = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        st.info(f"Embedding modeli: **{embed_model_name}** (Ollama) | Koleksiyon: **{collection_name}** | Qdrant: **{qdrant_url}**")

        if st.button("İndeksi oluştur / güncelle"):
            pdf_paths: List[Path] = []

            if use_default:
                pdf_paths.extend(default_pdfs)

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

        k = st.number_input(
            "Her soru için alınacak context chunk sayısı (k)",
            min_value=1,
            max_value=20,
            value=5,
        )

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

            try:
                with st.spinner("Pipeline çalışıyor (RAG + değerlendirme)..."):
                    rows = run_full_pipeline(
                        csv_path=str(csv_path),
                        collection_name=collection_name,
                        qdrant_url=qdrant_url,
                        eval_model=eval_model_name,
                        k=int(k),
                        openai_client=client,
                        eval_backend="openai" if eval_backend == "OpenAI" else "ollama",
                        eval_local_model=local_eval_model_name,
                    )
            except Exception as exc:
                st.error(f"Pipeline çalışırken bir hata oluştu ve işlem durduruldu: {exc}")
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

        if "chat_eval_rows" not in st.session_state:
            st.session_state["chat_eval_rows"] = []

        question = st.text_area(
            "Soru",
            placeholder="Buraya modelden cevap almak istediğin soruyu yaz...",
            height=120,
        )

        expected_answer = st.text_area(
            "İsteğe bağlı: Beklenen / referans cevap",
            placeholder="Eval sırasında kıyaslamak için isteğe bağlı olarak doğru cevabı yazabilirsin.",
            height=120,
        )

        compare_no_rag = st.toggle(
            "RAG'li cevaba ek olarak RAG'siz cevabı da üret ve değerlendir",
            value=True,
        )

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
                compare_no_rag=compare_no_rag,
                k=k_chat,
                qa_models_selected=qa_models_selected,
                all_models=all_models,
                eval_backend=eval_backend,
                eval_model_name=eval_model_name,
                local_eval_model_name=local_eval_model_name,
                openai_api_key=openai_api_key,
                chroma_dir=chroma_dir,
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

    # ── 4th Tab: Sesli Değerlendirme (Voice) ──────────────────────────
    with tab_voice:
        st.subheader("Sesli Değerlendirme (RAG + TTS)")

        # --- Section 1: Soru Metni ---
        transcription = st.text_area(
            "Sorunuzu yazın",
            height=120,
            key="voice_transcription_area",
        )

        # --- Section 3: RAG Pipeline ---
        voice_expected = st.text_area(
            "İsteğe bağlı: Beklenen / referans cevap",
            placeholder="Eval sırasında kıyaslamak için isteğe bağlı olarak doğru cevabı yazabilirsin.",
            height=100,
            key="voice_expected_answer",
        )

        voice_compare_no_rag = st.toggle(
            "RAG'siz cevabı da üret",
            value=False,
            key="voice_no_rag_toggle",
        )

        voice_k = st.number_input(
            "RAG chunk sayısı (k)",
            min_value=1,
            max_value=20,
            value=5,
            key="voice_k",
        )

        if st.button("Soruyu Değerlendir", key="voice_eval_btn"):
            q = transcription.strip() # type: ignore
            if not q:
                st.error("Lütfen bir soru yazın.")
            else:
                answers = _run_chat_eval(
                    question=q,
                    expected_answer=voice_expected,
                    compare_no_rag=voice_compare_no_rag,
                    k=voice_k,
                    qa_models_selected=qa_models_selected,
                    all_models=all_models,
                    eval_backend=eval_backend,
                    eval_model_name=eval_model_name,
                    local_eval_model_name=local_eval_model_name,
                    openai_api_key=openai_api_key,
                    chroma_dir=chroma_dir,
                    collection_name=collection_name,
                )

                # --- Section 4: TTS Output ---
                if answers:
                    st.markdown("---")
                    st.markdown("### 🔊 Sesli Yanıt (TTS)")
                    for ans in answers:
                        label = f"{ans['model']} ({ans['mode']})"
                        answer_text = ans["answer"]
                        if not answer_text.strip():
                            continue
                        st.markdown(f"**{label}**")
                        with st.spinner(f"{label} için metin sese çevriliyor..."):
                            wav_bytes, sr = synthesize_speech(answer_text)
                        st.audio(wav_bytes, format="audio/wav")
                        st.download_button(
                            f"İndir — {label}",
                            data=wav_bytes,
                            file_name=f"cevap_{ans['model']}_{ans['mode']}.wav",
                            mime="audio/wav",
                            key=f"voice_dl_{ans['model']}_{ans['mode']}",
                        )


if __name__ == "__main__":
    main()
