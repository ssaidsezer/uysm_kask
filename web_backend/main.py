"""FastAPI application — full parity with streamlit_app.py tabs."""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Any, Callable, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from pipeline import (
    EVAL_MODEL_NAME,
    QA_OLLAMA_MODEL,
    get_openai_client,
    run_full_pipeline,
    unload_model,
    write_results_to_csv,
)
from rag_index import (
    DEFAULT_COLLECTION_NAME,
    QDRANT_URL,
    SMART_BOUNDARY_LLM_MODEL,
    SMART_CHILD_OVERLAP,
    SMART_CHILD_SIZE,
    SMART_PARENT_BLOCK_SIZE,
    index_pdfs,
    index_pdfs_smart,
)

from web_backend.analysis_service import analyze_export_csv
from web_backend.chat_service import run_chat_eval
from web_backend.compat import (
    collection_name_full,
    collection_options_for_embed,
    delete_ollama_model,
    delete_qdrant_collection,
    ensure_tmp_dir,
    get_monitor_stats,
    get_ollama_base_url,
    list_all_ollama_model_names_raw,
    list_embedding_models,
    list_ollama_models,
    list_qdrant_collections,
    pull_ollama_model,
    smart_collection_name_full,
)
from web_backend.jobs import job_store
from web_backend.schemas import (
    AnalysisResponse,
    AppConfigResponse,
    ChatEvalRequest,
    ChatEvalResponse,
    CollectionsResponse,
    ConnectionStatusResponse,
    DeleteModelRequest,
    EmbeddingModelsResponse,
    HealthResponse,
    IndexJobStartResponse,
    JobProgress,
    JobStatusResponse,
    OllamaModelsResponse,
    PullModelRequest,
    SimpleMessageResponse,
    TTSModelsResponse,
    TTSRequest,
)

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(WORKSPACE_DIR / ".env", override=True)

app = FastAPI(title="RAG Pipeline API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(
        ","
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _json_safe_rows(rows: List[dict]) -> List[dict]:
    out: List[dict] = []
    for r in rows:
        row = {}
        for k, v in r.items():
            if v is None or isinstance(v, (str, int, float, bool)):
                row[k] = v
            elif hasattr(v, "item"):  # numpy scalar
                row[k] = v.item()
            else:
                row[k] = str(v)
        out.append(row)
    return out


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/api/config", response_model=AppConfigResponse)
def config() -> AppConfigResponse:
    return AppConfigResponse(
        qdrant_url=os.environ.get("QDRANT_URL", QDRANT_URL),
        default_collection_name=os.environ.get("QDRANT_COLLECTION", DEFAULT_COLLECTION_NAME),
        ollama_base_url=get_ollama_base_url(),
        monitor_url=os.environ.get("MONITOR_URL", "http://192.168.1.151:8081"),
        qa_ollama_model=QA_OLLAMA_MODEL,
        eval_model_name=EVAL_MODEL_NAME,
        chunk_size_default=int(os.environ.get("CHUNK_SIZE", "1000")),
        chunk_overlap_default=int(os.environ.get("CHUNK_OVERLAP", "200")),
        smart_parent_size=SMART_PARENT_BLOCK_SIZE,
        smart_child_size=SMART_CHILD_SIZE,
        smart_child_overlap=SMART_CHILD_OVERLAP,
        smart_boundary_llm_model=SMART_BOUNDARY_LLM_MODEL,
    )


@app.get("/api/connection-status", response_model=ConnectionStatusResponse)
def connection_status() -> ConnectionStatusResponse:
    q_url = os.environ.get("QDRANT_URL", QDRANT_URL)
    q_ok, q_msg = True, f"Qdrant bağlı ({q_url})"
    try:
        import requests

        r = requests.get(f"{q_url}/collections", timeout=5)
        r.raise_for_status()
    except Exception:
        q_ok = False
        q_msg = f"Qdrant erişilemiyor ({q_url})"

    o_base = get_ollama_base_url()
    o_ok, o_msg = False, "Ollama tanımlı değil"
    if o_base:
        try:
            import requests

            r = requests.get(f"{o_base}/api/tags", timeout=5)
            r.raise_for_status()
            o_ok = True
            o_msg = f"Ollama bağlı ({o_base})"
        except Exception:
            o_msg = f"Ollama erişilemiyor ({o_base})"

    stats, m_err = get_monitor_stats()
    return ConnectionStatusResponse(
        qdrant_ok=q_ok,
        qdrant_message=q_msg,
        ollama_ok=o_ok,
        ollama_message=o_msg,
        monitor=stats,
        monitor_error=m_err if not stats else None,
    )


@app.get("/api/models/ollama", response_model=OllamaModelsResponse)
def api_ollama_models() -> OllamaModelsResponse:
    models, err, elapsed, fcount = list_ollama_models()
    return OllamaModelsResponse(
        models=sorted({m for m in models if m}),
        error=err,
        filter_elapsed_seconds=elapsed,
        filtered_embedding_count=fcount,
    )


@app.get("/api/models/embeddings", response_model=EmbeddingModelsResponse)
def api_embed_models() -> EmbeddingModelsResponse:
    models, err = list_embedding_models()
    return EmbeddingModelsResponse(models=models, error=err)


@app.get("/api/qdrant/collections", response_model=CollectionsResponse)
def api_collections() -> CollectionsResponse:
    cols, err = list_qdrant_collections()
    return CollectionsResponse(collections=cols, error=err)


@app.get("/api/collection-options")
def api_collection_options(embed_model: str) -> dict:
    cols, err = list_qdrant_collections()
    if err:
        return {"classic": [], "smart_bases": [], "error": err}
    classic, smart = collection_options_for_embed(cols, embed_model)
    return {"classic": classic, "smart_bases": smart, "error": ""}


@app.delete("/api/qdrant/collections/{name}", response_model=SimpleMessageResponse)
def api_delete_collection(name: str) -> SimpleMessageResponse:
    ok, msg = delete_qdrant_collection(name)
    return SimpleMessageResponse(success=ok, message=msg)


@app.post("/api/ollama/models/pull", response_model=SimpleMessageResponse)
def api_pull(req: PullModelRequest) -> SimpleMessageResponse:
    ok, msg = pull_ollama_model(req.name.strip())
    return SimpleMessageResponse(success=ok, message=msg)


@app.delete("/api/ollama/models", response_model=SimpleMessageResponse)
def api_delete_model(req: DeleteModelRequest) -> SimpleMessageResponse:
    ok, msg = delete_ollama_model(req.name.strip())
    return SimpleMessageResponse(success=ok, message=msg)


@app.get("/api/ollama/models/all-raw")
def api_all_ollama_raw() -> dict:
    names, err = list_all_ollama_model_names_raw()
    return {"models": names, "error": err}


# --- Index job ---


def _run_index_job(
    pdf_paths: List[str],
    *,
    use_smart: bool,
    collection_base: str,
    qdrant_url: str,
    embed_model: str,
    chunk_size: int,
    chunk_overlap: int,
    smart_parent_size: int,
    smart_child_size: int,
    smart_child_overlap: int,
    boundary_llm_model: str,
    progress_cb: Any,
) -> dict:
    if use_smart:
        index_name = smart_collection_name_full(
            collection_base,
            embed_model,
            smart_parent_size,
            smart_child_size,
            smart_child_overlap,
        )
        return index_pdfs_smart(
            pdf_paths,
            base_collection=index_name,
            parent_size=smart_parent_size,
            child_size=smart_child_size,
            child_overlap=smart_child_overlap,
            boundary_llm_model=boundary_llm_model,
            qdrant_url=qdrant_url,
            embed_model=embed_model,
            progress_callback=progress_cb,
        )

    index_name = collection_name_full(collection_base, embed_model, chunk_size, chunk_overlap)
    return index_pdfs(
        pdf_paths,
        collection_name=index_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        qdrant_url=qdrant_url,
        embed_model=embed_model,
        progress_callback=progress_cb,
    )


@app.post("/api/jobs/index", response_model=IndexJobStartResponse)
async def start_index_job(
    files: List[UploadFile] = File(...),
    use_smart: bool = Form(False),
    embed_model: str = Form(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    smart_parent_size: int = Form(SMART_PARENT_BLOCK_SIZE),
    smart_child_size: int = Form(SMART_CHILD_SIZE),
    smart_child_overlap: int = Form(SMART_CHILD_OVERLAP),
    boundary_llm_model: str = Form(SMART_BOUNDARY_LLM_MODEL),
) -> IndexJobStartResponse:
    if not files or not any(f.filename for f in files):
        raise HTTPException(status_code=400, detail="İndekslenecek PDF bulunamadı.")

    tmp = ensure_tmp_dir()
    paths: List[str] = []
    for up in files:
        if not up.filename or not up.filename.lower().endswith(".pdf"):
            continue
        dest = tmp / up.filename
        content = await up.read()
        dest.write_bytes(content)
        paths.append(str(dest))

    if not paths:
        raise HTTPException(status_code=400, detail="Geçerli PDF dosyası yok.")

    collection_base = os.environ.get("QDRANT_COLLECTION", DEFAULT_COLLECTION_NAME)
    qdrant_url = os.environ.get("QDRANT_URL", QDRANT_URL)

    jid = job_store.create()

    def run(progress: Callable[..., Any]) -> dict:
        return _run_index_job(
            paths,
            use_smart=use_smart,
            collection_base=collection_base,
            qdrant_url=qdrant_url,
            embed_model=embed_model,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            smart_parent_size=int(smart_parent_size),
            smart_child_size=int(smart_child_size),
            smart_child_overlap=int(smart_child_overlap),
            boundary_llm_model=boundary_llm_model,
            progress_cb=progress,
        )

    job_store.run_async(jid, run)
    return IndexJobStartResponse(job_id=jid)


@app.post("/api/jobs/csv-pipeline", response_model=IndexJobStartResponse)
async def start_csv_job(
    csv_file: Optional[UploadFile] = File(None),
    use_sample: bool = Form(False),
    eval_enabled: bool = Form(True),
    eval_backend: str = Form("OpenAI"),
    eval_model_name: str = Form(""),
    local_eval_model_name: Optional[str] = Form(None),
    csv_question_col: str = Form("question"),
    csv_answer_col: str = Form("answer"),
    csv_embed_model: Optional[str] = Form(None),
    csv_collection_name: str = Form(""),
    csv_smart_rag: bool = Form(False),
    csv_retrieval_mode: str = Form("vector"),
    csv_score_threshold: float = Form(0.55),
    rag_mode: str = Form("rag"),
    k: int = Form(5),
    thinking_enabled: bool = Form(False),
    qa_models_json: str = Form("[]"),
    openai_api_key: Optional[str] = Form(None),
) -> IndexJobStartResponse:
    """qa_models_json: JSON array of model names."""

    try:
        qa_models: List[str] = json.loads(qa_models_json) if qa_models_json else []
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="qa_models_json geçersiz JSON.")

    tmp = ensure_tmp_dir()
    sample_path = WORKSPACE_DIR / "sample_rag_input.csv"
    csv_path: Optional[Path] = None
    errors_pre: List[str] = []

    if csv_file and csv_file.filename:
        dest = tmp / "uploaded_input.csv"
        dest.write_bytes(await csv_file.read())
        csv_path = dest
    elif use_sample and sample_path.exists():
        csv_path = sample_path
    else:
        errors_pre.append("CSV seçilmedi.")

    client = None
    if csv_path is not None and eval_enabled and eval_backend == "OpenAI":
        key = (openai_api_key or "").strip() or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            errors_pre.append("OpenAI değerlendirme motoru seçili. OpenAI API key gerekli.")
            csv_path = None
        else:
            client = get_openai_client(api_key=key)

    if rag_mode != "no_rag" and not (csv_collection_name or "").strip() and not errors_pre:
        errors_pre.append("Seçili retrieval tipi için geçerli bir koleksiyon gerekli.")
        csv_path = None

    qdrant_url = os.environ.get("QDRANT_URL", QDRANT_URL)
    ollama_models, _, _, _ = list_ollama_models()
    all_m = sorted({m for m in ollama_models if m})

    jid = job_store.create()

    def run(progress: Callable[..., Any]) -> dict:
        rows: List[dict] = []
        errors: List[str] = list(errors_pre)

        if csv_path is None:
            return {"rows": [], "errors": errors}

        _models = qa_models or ([QA_OLLAMA_MODEL] if not all_m else [all_m[0]])
        n_models = len(_models)
        for idx, qa_model in enumerate(_models):
            try:
                model_rows = run_full_pipeline(
                    csv_path=str(csv_path),
                    collection_name=csv_collection_name,
                    qdrant_url=qdrant_url,
                    eval_model=eval_model_name or EVAL_MODEL_NAME,
                    k=int(k),
                    openai_client=client,
                    eval_backend="openai" if eval_backend == "OpenAI" else "ollama",
                    eval_local_model=local_eval_model_name,
                    qa_model=qa_model,
                    rag_mode=rag_mode,
                    eval_enabled=eval_enabled,
                    question_col=csv_question_col,
                    answer_col=csv_answer_col,
                    embed_model=csv_embed_model or "",
                    think=thinking_enabled,
                    smart_chunking=csv_smart_rag,
                    score_threshold=float(csv_score_threshold),
                    retrieval_mode=csv_retrieval_mode,
                )
                rows.extend(model_rows)
            except Exception as exc:
                errors.append(f"{qa_model} için pipeline hatası: {exc}")
            finally:
                unload_model(qa_model)
            progress(
                "csv_model",
                idx + 1,
                n_models,
                0.0,
            )

        buf = io.StringIO()
        write_results_to_csv(rows, buf)
        return {
            "rows": _json_safe_rows(rows),
            "errors": errors,
            "csv_text": buf.getvalue(),
        }

    job_store.run_async(jid, run)
    return IndexJobStartResponse(job_id=jid)


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    data = job_store.get(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job bulunamadı.")

    prog = data.get("progress")
    progress_model = None
    if prog:
        progress_model = JobProgress(
            phase=str(prog.get("phase", "")),
            current=int(prog.get("current", 0)),
            total=int(prog.get("total", 0)),
            elapsed_sec=float(prog.get("elapsed_sec", 0.0)),
        )

    return JobStatusResponse(
        id=job_id,
        status=data["status"],
        progress=progress_model,
        result=data.get("result"),
        error=data.get("error"),
    )


@app.post("/api/chat/eval", response_model=ChatEvalResponse)
def chat_eval(req: ChatEvalRequest) -> ChatEvalResponse:
    models, err, _, _ = list_ollama_models()
    if err:
        return ChatEvalResponse(errors=[err])
    all_models = sorted({m for m in models if m})
    try:
        return run_chat_eval(req, all_models)
    except ValueError as exc:
        return ChatEvalResponse(errors=[str(exc)])
    except Exception as exc:
        return ChatEvalResponse(
            errors=[f"Manual chat eval sırasında beklenmeyen bir hata oluştu: {exc}"]
        )


@app.get("/api/voice/models", response_model=TTSModelsResponse)
def voice_models() -> TTSModelsResponse:
    from voice_utils import get_downloaded_tts_models

    return TTSModelsResponse(downloaded_models=get_downloaded_tts_models())


@app.post("/api/voice/tts")
def voice_tts(req: TTSRequest) -> Response:
    from voice_utils import synthesize_speech

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Metin boş.")
    try:
        wav_bytes, _sr, duration = synthesize_speech(
            text,
            model=req.model,
            speaker_id=req.speaker_id,
            voice_preset=req.voice_preset if req.voice_preset != "Varsayılan" else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"X-Audio-Duration": str(duration)},
    )


@app.post("/api/analysis", response_model=AnalysisResponse)
async def analysis(file: UploadFile = File(...)) -> AnalysisResponse:
    content = await file.read()
    return analyze_export_csv(content)
