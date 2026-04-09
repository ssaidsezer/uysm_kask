"""Manual chat evaluation — retrieval, QA, and scoring; returns JSON for the API."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import List

from pipeline import (
    EVAL_MODEL_NAME,
    evaluate_answer_any,
    generate_no_rag_answer_ollama,
    generate_rag_answer_ollama,
    get_openai_client,
    warmup_model,
    unload_model,
)
from rag_index import (
    retrieve_chunks,
    retrieve_chunks_bm25,
    retrieve_chunks_bm25_smart,
    retrieve_chunks_smart,
)

from .model_profiles_store import resolve_eval_profile, resolve_qa_for_model
from .schemas import ChatEvalRequest, ChatEvalResponse, ChatModelResult, ChunkCard


def _rag_answer_record(qa_model_name: str, req: ChatEvalRequest, rag_result: dict) -> dict:
    return {
        "model": f"{qa_model_name} (RAG)" if req.rag_mode == "both" else qa_model_name,
        "question_index": 0,
        "question": req.question,
        "observation_idea": req.expected_answer or "",
        "model_answer": rag_result.get("answer", ""),
        "response_time_seconds": rag_result.get("response_time_seconds", 0.0),
    }


def _no_rag_answer_record(qa_model_name: str, req: ChatEvalRequest, no_rag_result: dict) -> dict:
    return {
        "model": f"{qa_model_name} (RAG'siz)" if req.rag_mode == "both" else qa_model_name,
        "question_index": 0,
        "question": req.question,
        "observation_idea": req.expected_answer or "",
        "model_answer": no_rag_result.get("answer", ""),
        "response_time_seconds": no_rag_result.get("response_time_seconds", 0.0),
    }


def _evaluate_record(
    req: ChatEvalRequest,
    record: dict,
    openai_client,
    eval_model: str,
    eval_sys,
    eval_oopts,
    eval_think_prof,
    eval_openai_extras,
) -> dict:
    if req.eval_enabled:
        return evaluate_answer_any(
            record=record,
            eval_model=eval_model,
            client=openai_client,
            backend="openai" if req.eval_backend == "OpenAI" else "ollama",
            local_model=req.local_eval_model_name,
            eval_system_prompt=eval_sys,
            eval_ollama_options=eval_oopts or None,
            eval_think=eval_think_prof,
            eval_openai_extras=eval_openai_extras or None,
        )
    return {}


def _try_rag_answer_pipeline(
    *,
    req: ChatEvalRequest,
    qa_model_name: str,
    qa_sys,
    qa_opts,
    qa_think: bool,
    context: str,
    openai_client,
    eval_model: str,
    eval_sys,
    eval_oopts,
    eval_think_prof,
    eval_openai_extras,
) -> tuple[dict | None, dict | None, dict | None, str | None]:
    try:
        rag_result = generate_rag_answer_ollama(
            question=req.question,
            context=context,
            model=qa_model_name,
            system_prompt=qa_sys,
            ollama_options=qa_opts or None,
            think=qa_think,
        )
        rag_record = _rag_answer_record(qa_model_name, req, rag_result)
        rag_eval = _evaluate_record(
            req,
            rag_record,
            openai_client,
            eval_model,
            eval_sys,
            eval_oopts,
            eval_think_prof,
            eval_openai_extras,
        )
        return rag_record, rag_result, rag_eval, None
    except Exception as exc:
        return None, None, None, f"RAG'li çağrıda hata: {exc}"


def _try_no_rag_answer_pipeline(
    *,
    req: ChatEvalRequest,
    qa_model_name: str,
    qa_sys,
    qa_opts,
    qa_think: bool,
    openai_client,
    eval_model: str,
    eval_sys,
    eval_oopts,
    eval_think_prof,
    eval_openai_extras,
) -> tuple[dict | None, dict | None, dict | None, str | None]:
    try:
        no_rag_result = generate_no_rag_answer_ollama(
            question=req.question,
            model=qa_model_name,
            system_prompt=qa_sys,
            ollama_options=qa_opts or None,
            think=qa_think,
        )
        no_rag_record = _no_rag_answer_record(qa_model_name, req, no_rag_result)
        no_rag_eval = _evaluate_record(
            req,
            no_rag_record,
            openai_client,
            eval_model,
            eval_sys,
            eval_oopts,
            eval_think_prof,
            eval_openai_extras,
        )
        return no_rag_record, no_rag_result, no_rag_eval, None
    except Exception as exc:
        return None, None, None, f" RAG'siz çağrıda hata: {exc}"


def _chunk_title(retrieval_mode: str, smart_rag: bool) -> str:
    if retrieval_mode == "bm25" and smart_rag:
        return (
            "Retrieved Chunks — BM25 Smart "
            "(BM25 ile child arama, bağlam olarak parent blok kullanılır)"
        )
    if retrieval_mode == "bm25":
        return "Retrieved Chunks — BM25 Klasik (anahtar kelime tabanlı BM25 arama)"
    if smart_rag:
        return (
            "Retrieved Chunks — Smart RAG "
            "(kartlar eşleşen child chunk'ı, bağlam olarak parent blok kullanılır)"
        )
    return "Retrieved Chunks"


def run_chat_eval(req: ChatEvalRequest, all_models: List[str]) -> ChatEvalResponse:
    errors: List[str] = []
    chat_eval_rows: List[dict] = []

    if req.eval_enabled and req.eval_backend == "OpenAI":
        key = (req.openai_api_key or "").strip() or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            return ChatEvalResponse(
                errors=[
                    "OpenAI değerlendirme motoru seçili. OpenAI API key gerekli "
                    "(istek gövdesinde veya OPENAI_API_KEY ortam değişkeni)."
                ]
            )
        openai_client = get_openai_client(api_key=key)
    else:
        openai_client = None

    retrieved_chunks_list: List[dict] = []
    context = ""
    if req.rag_mode in ("rag", "both"):
        if req.retrieval_mode == "bm25":
            if req.smart_rag:
                retrieved_chunks_list = retrieve_chunks_bm25_smart(
                    question=req.question,
                    base_collection=req.collection_name,
                    k=int(req.k),
                )
            else:
                retrieved_chunks_list = retrieve_chunks_bm25(
                    question=req.question,
                    collection_name=req.collection_name,
                    k=int(req.k),
                )
        elif req.smart_rag:
            retrieved_chunks_list = retrieve_chunks_smart(
                question=req.question,
                base_collection=req.collection_name,
                k=int(req.k),
                embed_model=req.embed_model or None,
                score_threshold=req.score_threshold,
            )
        else:
            retrieved_chunks_list = retrieve_chunks(
                question=req.question,
                collection_name=req.collection_name,
                k=int(req.k),
                embed_model=req.embed_model or None,
                score_threshold=req.score_threshold,
            )
        context = "\n\n".join(c["text"] for c in retrieved_chunks_list)

    chunk_warning: str | None = None
    section_title = ""
    chunk_cards: List[ChunkCard] = []

    if req.rag_mode in ("rag", "both") and retrieved_chunks_list:
        section_title = _chunk_title(req.retrieval_mode, req.smart_rag)
        for c in retrieved_chunks_list:
            chunk_cards.append(
                ChunkCard(
                    text=c["text"],
                    score=float(c["score"]),
                    child_text=c.get("child_text"),
                )
            )
    elif req.rag_mode in ("rag", "both"):
        if req.retrieval_mode == "bm25":
            chunk_warning = (
                "BM25 araması herhangi bir chunk döndürmedi. "
                "Koleksiyonun dolu olduğundan emin olun."
            )
        else:
            chunk_warning = (
                f"Qdrant'ta bu sorgu için score_threshold={req.score_threshold:.2f} "
                "üstünde chunk bulunamadı. Threshold'u düşürmeyi deneyin."
            )

    run_timestamp = datetime.now(timezone.utc).isoformat()
    selected = req.qa_models_selected or all_models
    model_results: List[ChatModelResult] = []

    eval_model = req.eval_model_name or EVAL_MODEL_NAME

    eval_sys, eval_oopts, eval_think_prof, eval_openai_extras, eval_trace = resolve_eval_profile(
        eval_backend=req.eval_backend,
        eval_model_name=eval_model,
        local_eval_model_name=req.local_eval_model_name,
        use_saved_defaults=req.use_saved_eval_defaults,
        profile_id_override=req.eval_profile_id,
        param_overrides=req.eval_param_overrides,
    )

    for qa_model_name in selected:
        warmup_model(model=qa_model_name)
        pid_override = (req.qa_profile_by_model or {}).get(qa_model_name)
        qa_sys, qa_opts, qa_think_prof, qa_trace = resolve_qa_for_model(
            qa_model_name,
            use_saved_defaults=req.use_saved_qa_defaults,
            profile_id_override=pid_override or None,
            param_overrides=req.qa_param_overrides,
        )
        qa_think = bool(req.thinking_enabled or qa_think_prof)
        trace = {**qa_trace, **eval_trace}
        block = ChatModelResult(model_name=qa_model_name, profile_trace=trace or None)
        rag_record = rag_result = rag_eval = None
        no_rag_record = no_rag_result = no_rag_eval = None

        if req.rag_mode in ("rag", "both"):
            rag_record, rag_result, rag_eval, rag_err = _try_rag_answer_pipeline(
                req=req,
                qa_model_name=qa_model_name,
                qa_sys=qa_sys,
                qa_opts=qa_opts,
                qa_think=qa_think,
                context=context,
                openai_client=openai_client,
                eval_model=eval_model,
                eval_sys=eval_sys,
                eval_oopts=eval_oopts,
                eval_think_prof=eval_think_prof,
                eval_openai_extras=eval_openai_extras,
            )
            if rag_err:
                block.error = rag_err
                model_results.append(block)
                unload_model(qa_model_name)
                continue

        if req.rag_mode in ("no_rag", "both"):
            no_rag_record, no_rag_result, no_rag_eval, no_rag_err = _try_no_rag_answer_pipeline(
                req=req,
                qa_model_name=qa_model_name,
                qa_sys=qa_sys,
                qa_opts=qa_opts,
                qa_think=qa_think,
                openai_client=openai_client,
                eval_model=eval_model,
                eval_sys=eval_sys,
                eval_oopts=eval_oopts,
                eval_think_prof=eval_think_prof,
                eval_openai_extras=eval_openai_extras,
            )
            if no_rag_err:
                block.error = (block.error or "") + no_rag_err
                model_results.append(block)
                unload_model(qa_model_name)
                continue

        if req.rag_mode == "both":
            if not rag_record or not no_rag_record or not rag_result or not no_rag_result:
                block.error = "Çift mod çıktısı eksik (RAG veya RAG'siz tamamlanamadı)."
                model_results.append(block)
                unload_model(qa_model_name)
                continue

        block.rag_answer = rag_record["model_answer"] if rag_record else None
        block.rag_eval = rag_eval if rag_eval is not None else None
        block.rag_result_meta = _meta(rag_result) if rag_result else None
        block.no_rag_answer = no_rag_record["model_answer"] if no_rag_record else None
        block.no_rag_eval = no_rag_eval if no_rag_eval is not None else None
        block.no_rag_result_meta = _meta(no_rag_result) if no_rag_result else None
        model_results.append(block)

        if rag_record and rag_result:
            row_rag = {
                "timestamp": run_timestamp,
                "model": qa_model_name,
                "mode": "RAG",
                "question": req.question,
                "expected_answer": (req.expected_answer or "").strip(),
                "model_answer": rag_record["model_answer"],
                "response_time_seconds": rag_record["response_time_seconds"],
                "tokens_per_second": rag_result.get("tokens_per_second") or "",
                "ai_score": (rag_eval or {}).get("ai_score", ""),
                "ai_verdict": (rag_eval or {}).get("ai_verdict", ""),
                "ai_hallucination_risk": (rag_eval or {}).get("ai_hallucination_risk", ""),
                "retrieved_chunks": json.dumps(
                    [c["text"] for c in retrieved_chunks_list], ensure_ascii=False
                ),
            }
            row_rag.update({k: v for k, v in trace.items() if v is not None})
            chat_eval_rows.append(row_rag)

        if no_rag_record and no_rag_result:
            row_nr = {
                "timestamp": run_timestamp,
                "model": qa_model_name,
                "mode": "NO_RAG",
                "question": req.question,
                "expected_answer": (req.expected_answer or "").strip(),
                "model_answer": no_rag_record["model_answer"],
                "response_time_seconds": no_rag_record["response_time_seconds"],
                "eval_duration_seconds": no_rag_result.get("eval_duration_seconds") or "",
                "tokens_per_second": no_rag_result.get("tokens_per_second") or "",
                "ai_score": (no_rag_eval or {}).get("ai_score", ""),
                "ai_verdict": (no_rag_eval or {}).get("ai_verdict", ""),
                "ai_hallucination_risk": (no_rag_eval or {}).get("ai_hallucination_risk", ""),
                "retrieved_chunks": "[]",
            }
            row_nr.update({k: v for k, v in trace.items() if v is not None})
            chat_eval_rows.append(row_nr)

        unload_model(qa_model_name)

    return ChatEvalResponse(
        chunks=chunk_cards,
        chunk_section_title=section_title,
        chunk_warning=chunk_warning,
        model_results=model_results,
        chat_eval_rows=chat_eval_rows,
        errors=errors,
    )


def _meta(result: dict | None) -> dict | None:
    if not result:
        return None
    return {
        "response_time_seconds": result.get("response_time_seconds"),
        "tokens_per_second": result.get("tokens_per_second"),
        "eval_duration_seconds": result.get("eval_duration_seconds"),
    }
