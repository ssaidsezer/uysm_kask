from __future__ import annotations

import csv
import io
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

# Reduce noisy HF tokenizers parallelism warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rag_index import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION_NAME,
    index_pdfs,
    load_embedding_model,
    get_chroma_client,
    get_or_create_collection,
)
from pipeline import (
    QA_OLLAMA_MODEL,
    EVAL_MODEL_NAME,
    evaluate_answer,
    generate_rag_answer_ollama,
    generate_no_rag_answer_ollama,
    get_openai_client,
    run_full_pipeline,
    write_results_to_csv,
)


WORKSPACE_DIR = Path(__file__).resolve().parent
load_dotenv(WORKSPACE_DIR / ".env")


def _list_ollama_models() -> List[str]:
    """
    List models from `ollama list` if available.
    """
    try:
        # Prefer JSON output if supported
        result = subprocess.run(
            ["ollama", "list", "--format", "json"],
            check=True,
            capture_output=True,
            text=True,
        )
        import json

        data = json.loads(result.stdout or "[]")
        models = []
        for item in data:
            name = item.get("name")
            if isinstance(name, str):
                models.append(name)
        return models
    except Exception:
        # Fallback to plain-text parsing
        try:
            result = subprocess.run(
                ["ollama", "list"],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return []

        models: List[str] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.lower().startswith("name"):
                continue
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models


def _list_lmstudio_models() -> List[str]:
    """
    Best-effort scan of LM Studio models directory on macOS.
    Varsayılan olarak ~/.lmstudio/models altına bakar.
    """
    # Tercih edilen dizini ortam değişkeniyle özelleştirebilme imkanı ver.
    base_env = os.environ.get("LMSTUDIO_MODELS_DIR")
    if base_env:
        base = Path(base_env).expanduser()
    else:
        # LM Studio settings.json içindeki "downloadsFolder" ile uyumlu varsayılan.
        base = Path.home() / ".lmstudio" / "models"

    if not base.exists():
        return []

    models: List[str] = []
    for root, _, files in os.walk(base):
        for fname in files:
            if fname.lower().endswith((".gguf", ".bin", ".safetensors")):
                rel = Path(root).joinpath(fname).relative_to(base)
                models.append(str(rel))
    return sorted(set(models))


def _list_hf_cache_models() -> List[str]:
    """
    List cached Hugging Face models from ~/.cache/huggingface/hub.
    """
    base = Path.home() / ".cache" / "huggingface" / "hub"
    if not base.exists():
        return []

    models: List[str] = []
    prefix = "models--"
    for entry in base.iterdir():
        if entry.is_dir() and entry.name.startswith(prefix):
            model_id = entry.name[len(prefix) :].replace("--", "/")
            models.append(model_id)
    return sorted(set(models))


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


def main() -> None:
    st.set_page_config(page_title="RAG Değerlendirme Pipeline", layout="wide")

    st.title("RAG + Değerlendirme Pipeline")
    st.write(
        "PDF tabanlı Türkçe RAG sistemi: "
        "PDF'leri Chroma'ya indeksle, CSV'den soruları değerlendir, "
        "cevapları Ollama (Qwen3) ile üret ve OpenAI ile puanla."
    )

    with st.sidebar:
        st.header("Ayarlar")

        # Bu üç ayar artık yalnızca ortam değişkenlerinden / kod sabitlerinden okunuyor.
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        chroma_dir = os.environ.get("CHROMA_DIR", DEFAULT_CHROMA_DIR)
        collection_name = os.environ.get("COLLECTION_NAME", DEFAULT_COLLECTION_NAME)

        # Yerel modelleri tara (Ollama, LM Studio, Hugging Face cache)
        with st.spinner("Yerel modeller taranıyor..."):
            ollama_models = _list_ollama_models()
            lmstudio_models = _list_lmstudio_models()
            hf_models = _list_hf_cache_models()

        all_models: List[str] = sorted(
            {m for m in (ollama_models + lmstudio_models + hf_models) if m}
        )
        if not all_models:
            all_models = [QA_OLLAMA_MODEL]

        # Değerlendirilecek QA modelleri: arama kutulu ve açılır-kapanır seçim listesi
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

        # Değerlendirme motoruna göre sadece ilgili model alanını göster
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
                options=all_models,
                index=0,
                help="Eval için kullanılacak yerel Ollama modelini seç.",
            )

    tab_index, tab_eval, tab_chat = st.tabs(
        ["📚 PDF İndeksleme", "📊 CSV Değerlendirme", "💬 Manuel Chat Eval"]
    )

    with tab_index:
        st.subheader("PDF'leri Chroma'ya indeksle")

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

        # Chunk ayarlarını artık sadece konfigürasyondan okuyalım (UI'da göstermiyoruz)
        chunk_size = int(os.environ.get("CHUNK_SIZE", "1000"))
        chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", "200"))

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
                with st.spinner("Embedding modeli yükleniyor..."):
                    embedding_model = load_embedding_model()

                client = get_chroma_client(persist_directory=chroma_dir)
                collection = get_or_create_collection(client, name=collection_name)

                with st.spinner("PDF'ler indeksleniyor..."):
                    total_chunks = index_pdfs(
                        [str(p) for p in pdf_paths],
                        collection=collection,
                        embedding_model=embedding_model,
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                    )

                st.success(f"İndeksleme tamamlandı. Toplam chunk sayısı: {total_chunks}.")
                st.write("Koleksiyon:", collection_name)

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
                        "OpenAI değerlendirme motoru seçili. OpenAI API key gerekli (sidebar'dan girin veya ortam değişkeni ayarlayın)."
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
                        chroma_dir=chroma_dir,
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
            _ = write_results_to_csv(rows, output_path=output_csv)  # type: ignore[arg-type]

            csv_bytes = output_csv.getvalue().encode("utf-8")
            st.download_button(
                label="Sonuç CSV'yi indir",
                data=csv_bytes,
                file_name="output.csv",
                mime="text/csv",
            )

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

            if eval_backend == "OpenAI":
                if not openai_api_key and not os.environ.get("OPENAI_API_KEY"):
                    st.error(
                        "OpenAI değerlendirme motoru seçili. OpenAI API key gerekli (sidebar'dan girin veya ortam değişkeni ayarlayın)."
                    )
                    return
                openai_client = get_openai_client(api_key=openai_api_key or None)
            else:
                openai_client = None

            # Hazır bir Chroma koleksiyonu ve embedding modeli yükle
            client = get_chroma_client(persist_directory=chroma_dir)
            collection = get_or_create_collection(client, name=collection_name)
            embedding_model = load_embedding_model()

            # Ortak RAG context'ini bir kez hesapla
            context = ""
            try:
                from rag_index import retrieve_context  # local import to avoid cycle
            except Exception:
                retrieve_context = None  # type: ignore[assignment]

            if retrieve_context is not None:
                context = retrieve_context(
                    question=question,
                    collection=collection,
                    embedding_model=embedding_model,
                    k=int(k_chat),
                )

            run_timestamp = datetime.utcnow().isoformat()
            selected_models = qa_models_selected or all_models

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
                    # Bu model için hem RAG'li hem RAG'siz değerlendirmeyi atla
                    continue

                with cols[0]:
                    st.markdown("**RAG'li cevap (Ollama)**")
                    st.write(rag_record["model_answer"])
                    st.markdown("**RAG'li cevap eval (OpenAI)**")
                    st.json(rag_eval)

                # RAG'li satırı CSV loguna ekle
                st.session_state["chat_eval_rows"].append(
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

                    # RAG'siz satırı CSV loguna ekle
                    st.session_state["chat_eval_rows"].append(
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


if __name__ == "__main__":
    main()

