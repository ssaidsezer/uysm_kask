from __future__ import annotations

import csv
import json
import os
import time
from typing import Dict, List, Optional, Sequence, TextIO, Union

import requests
from openai import OpenAI

from rag_index import (
    DEFAULT_COLLECTION_NAME,
    QDRANT_URL,
    get_qdrant_client,
    retrieve_chunks,
)


QA_OLLAMA_MODEL = os.getenv("QA_OLLAMA_MODEL", "qwen3:1.7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "")
EVAL_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", "gpt-5.4-mini")


def load_questions(
    csv_path: str,
    question_col: str = "question",
    answer_col: str = "answer",
) -> List[Dict]:
    """
    Load questions and optional observation_idea from a CSV.
    Column names are configurable via question_col and answer_col parameters.
    """
    items: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            question = (row.get(question_col) or "").strip()
            observation = (row.get(answer_col) or "").strip()
            if not question:
                continue
            items.append(
                {
                    "question_index": idx,
                    "question": question,
                    "observation_idea": observation,
                }
            )
    return items


def _build_rag_prompt(question: str, context: str) -> str:
    """
    Basic Turkish RAG prompt: answer only from given context.
    """
    if not context.strip():
        return (
            "Aşağıdaki soruyu cevaplamaya çalışıyorsun, ancak sana hiçbir bağlam verilmiyor.\n"
            "Cevabı bilmiyorsan 'BİLMİYORUM' de ve uydurma.\n\n"
            f"Soru: {question}\n"
        )

    return (
        "Sana verilen metni bağlam olarak kullanarak soruyu cevapla.\n"
        "Kurallar:\n"
        "- Sadece aşağıdaki bağlamdaki bilgilere dayan.\n"
        "- Bağlamda olmayan bilgileri uydurma.\n"
        "- Eğer bağlam soruyu cevaplamak için yeterli değilse kısaca 'BİLMİYORUM' de.\n\n"
        f"Soru:\n{question}\n\n"
        f"Bağlam:\n{context}\n\n"
        "Cevabın:\n"
    )

def warmup_model(
    model: str = QA_OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = 120,
) -> None:
    """
    Ollama'ya modeli RAM'e yüklemesi için boş bir istek gönderir.
    Bu çağrı response_time ölçümüne dahil edilmez.
    """
    if not base_url:
        return
    try:
        requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={"model": model, "keep_alive": "5m"},
            timeout=timeout,
        )
    except Exception:
        pass


def unload_model(
    model: str,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = 30,
) -> None:
    """
    keep_alive=0 göndererek modeli Ollama'nın VRAM/RAM'inden boşaltır.
    """
    if not base_url:
        return
    try:
        requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=timeout,
        )
    except Exception:
        pass


def _require_ollama_base_url(base_url: str) -> str:
    if not base_url:
        raise ValueError(
            "OLLAMA_BASE_URL ortam değişkeni tanımlı değil. "
            "Lütfen .env dosyasına uzak sunucu adresini ekleyin (örn: OLLAMA_BASE_URL=http://192.168.1.151:11434)."
        )
    return base_url.rstrip("/")


def _collect_ollama_chat_response(
    *,
    prompt: str,
    model: str,
    base_url: str,
    timeout: int,
    think: bool,
) -> Dict:
    api_base = _require_ollama_base_url(base_url)
    t0 = time.time()
    payload: Dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": think,
    }
    resp = requests.post(
        f"{api_base}/api/chat",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()

    chunks: list[str] = []
    thinking_chunks: list[str] = []
    last_chunk = resp.json() or {}
    message = last_chunk.get("message") or {}
    content = message.get("content") or ""
    thinking = message.get("thinking") or ""
    if content:
        chunks.append(content)
    if thinking:
        thinking_chunks.append(thinking)

    answer = "".join(chunks).replace("\r\n", "\n").strip()
    result = {
        "answer": answer,
        "thinking": "".join(thinking_chunks).replace("\r\n", "\n").strip() if think else "",
        "response_time_seconds": time.time() - t0,
        "eval_count": None,
        "eval_duration_seconds": None,
        "tokens_per_second": None,
    }
    raw_eval_count = last_chunk.get("eval_count")
    raw_eval_duration = last_chunk.get("eval_duration")
    if isinstance(raw_eval_count, (int, float)) and isinstance(raw_eval_duration, (int, float)):
        result["eval_count"] = int(raw_eval_count)
        result["eval_duration_seconds"] = float(raw_eval_duration) / 1e9
        if result["eval_duration_seconds"] > 0:
            result["tokens_per_second"] = result["eval_count"] / result["eval_duration_seconds"]
    return result


def generate_rag_answer_ollama(
    question: str,
    context: str,
    model: str = QA_OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = 120,
    think: bool = False,
) -> Dict:
    """
    Call a local Ollama model (e.g. Qwen3 1.7B) with question + context.

    Returns a dict with:
      - answer: str
      - response_time_seconds: float

    think: Reasoning modeli için thinking modunu aç/kapat (varsayılan kapalı).
           Kapalıyken trace temizlenir, eval sonuçları temiz kalır.
    """
    prompt = _build_rag_prompt(question, context)
    return _collect_ollama_chat_response(
        prompt=prompt,
        model=model,
        base_url=base_url,
        timeout=timeout,
        think=think,
    )


def _build_no_rag_prompt(question: str) -> str:
    """
    Simple Turkish non-RAG prompt: answer from general knowledge.
    """
    return (
        "Sen Türkçe konuşan bir uzmansın.\n"
        "Aşağıdaki soruyu kendi bilginle, net ve kısa biçimde cevapla.\n"
        "Cevaptan emin değilsen dürüst ol ve uydurma.\n\n"
        f"Soru: {question}\n\n"
        "Cevabın:\n"
    )


def generate_no_rag_answer_ollama(
    question: str,
    model: str = QA_OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = 120,
    think: bool = False,
) -> Dict:
    """
    Call a local Ollama model WITHOUT any retrieved context (no RAG).

    Returns a dict with:
      - answer: str
      - response_time_seconds: float

    think: Reasoning modeli için thinking modunu aç/kapat (varsayılan kapalı).
    """
    prompt = _build_no_rag_prompt(question)
    return _collect_ollama_chat_response(
        prompt=prompt,
        model=model,
        base_url=base_url,
        timeout=timeout,
        think=think,
    )


def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """
    Create an OpenAI client. If api_key is None, uses OPENAI_API_KEY env var.
    """
    if api_key is None:
        return OpenAI()
    return OpenAI(api_key=api_key)


def evaluate_answer(
    record: Dict,
    eval_model: str = EVAL_MODEL_NAME,
    client: Optional[OpenAI] = None,
) -> Dict:
    """
    Evaluate a single QA record with OpenAI, returning a flat dict suitable for CSV.

    record MUST contain:
      - model
      - question_index
      - question
      - observation_idea
      - model_answer
      - response_time_seconds
    """
    if client is None:
        client = get_openai_client()

    system = (
        "Sen taktik muharebe ve askeri eğitim alanında uzman bir otomatik değerlendirici modelsin.\n"
        "Aşağıdaki kurallara kesinlikle uy:\n"
        "- Yalnızca tek bir düz JSON nesnesi döndür; başka hiçbir metin ekleme.\n"
        "- Tüm string değerleri Türkçe yaz.\n"
        "- 'ai_verdict' için SADECE şu değerlerden birini kullan: correct | partial | incorrect\n"
        "  * correct: cevap beklenenle büyük ölçüde örtüşüyor, kritik bilgi eksik değil\n"
        "  * partial: cevap kısmen doğru fakat önemli detay eksik ya da yanıltıcı\n"
        "  * incorrect: cevap yanlış, tehlikeli veya tamamen alakasız\n"
        "- 'ai_score' 0-10 arası tam sayı olsun (10=mükemmel, 5=kısmen doğru, 0=yanlış/tehlikeli)\n"
        "- 'ai_hallucination_risk' için SADECE şu değerlerden birini kullan: low | medium | high\n"
        "- 'ai_reason' en fazla 2 kısa Türkçe cümle; neyin doğru/yanlış olduğunu açıkla\n\n"
        "Döndürülecek JSON şeması (başka alan ekleme):\n"
        "{\n"
        '  "ai_verdict": "correct" veya "partial" veya "incorrect",\n'
        '  "ai_score": 0-10 tam sayı,\n'
        '  "ai_hallucination_risk": "low" veya "medium" veya "high",\n'
        '  "ai_reason": string\n'
        "}\n"
    )

    user = (
        f'Soru: "{record["question"]}"\n'
        f'Beklenen cevap: "{record.get("observation_idea", "")}"\n'
        f'Modelin cevabı: "{record.get("model_answer", "")}"\n'
    )

    response = client.chat.completions.create(
        model=eval_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        reasoning_effort="medium",
    )

    msg = response.choices[0].message

    parsed: Dict
    content = msg.content or "{}"
    parsed = json.loads(content)

    for k, v in list(parsed.items()):
        if isinstance(v, str):
            parsed[k] = v.replace("\r\n", " ").replace("\n", " ").strip()

    return parsed


def _extract_json_from_text(text: str) -> str:
    """
    Yerel modeller bazen JSON dışında ek metin üretebilir.
    İlk '{' ile son '}' arasını almaya çalış.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _evaluate_answer_local(
    record: Dict,
    model: str,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = 120,
) -> Dict:
    """
    OpenAI yerine yerel bir modeli (örn. Ollama) eval için kullan.
    """
    system = (
        "Sen taktik muharebe ve askeri eğitim alanında uzman bir otomatik değerlendirici modelsin.\n"
        "Aşağıdaki kurallara kesinlikle uy:\n"
        "- Yalnızca tek bir düz JSON nesnesi döndür; başka hiçbir metin ekleme.\n"
        "- Tüm string değerleri Türkçe yaz.\n"
        "- 'ai_verdict' için SADECE şu değerlerden birini kullan: correct | partial | incorrect\n"
        "  * correct: cevap beklenenle büyük ölçüde örtüşüyor, kritik bilgi eksik değil\n"
        "  * partial: cevap kısmen doğru fakat önemli detay eksik ya da yanıltıcı\n"
        "  * incorrect: cevap yanlış, tehlikeli veya tamamen alakasız\n"
        "- 'ai_score' 0-10 arası tam sayı olsun (10=mükemmel, 5=kısmen doğru, 0=yanlış/tehlikeli)\n"
        "- 'ai_hallucination_risk' için SADECE şu değerlerden birini kullan: low | medium | high\n"
        "- 'ai_reason' en fazla 2 kısa Türkçe cümle; neyin doğru/yanlış olduğunu açıkla\n\n"
        "Döndürülecek JSON şeması (başka alan ekleme):\n"
        "{\n"
        '  \"ai_verdict\": \"correct\" veya \"partial\" veya \"incorrect\",\n'
        '  \"ai_score\": 0-10 tam sayı,\n'
        '  \"ai_hallucination_risk\": \"low\" veya \"medium\" veya \"high\",\n'
        '  \"ai_reason\": string\n'
        "}\n"
    )

    user = (
        f'Soru: "{record["question"]}"\n'
        f'Beklenen cevap: "{record.get("observation_idea", "")}"\n'
        f'Modelin cevabı: "{record.get("model_answer", "")}"\n'
    )

    prompt = (
        system
        + "\n\n"
        + user
        + "\n\nYukarıdaki talimatlara göre SADECE geçerli bir JSON nesnesi üret. Başka metin ekleme."
    )

    if not base_url:
        raise ValueError(
            "OLLAMA_BASE_URL ortam değişkeni tanımlı değil. "
            "Lütfen .env dosyasına uzak sunucu adresini ekleyin."
        )

    resp = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    content = ""
    if isinstance(data, dict):
        message = data.get("message") or {}
        if isinstance(message, dict):
            content = str(message.get("content") or "")
        elif "choices" in data:
            choices = data.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = str(msg.get("content") or "")

    content = content.replace("\r\n", "\n").strip()
    try:
        json_text = _extract_json_from_text(content)
        parsed = json.loads(json_text)
    except Exception:
        parsed = {}

    for k, v in list(parsed.items()):
        if isinstance(v, str):
            parsed[k] = v.replace("\r\n", " ").replace("\n", " ").strip()

    return parsed


def evaluate_answer_any(
    record: Dict,
    eval_model: str = EVAL_MODEL_NAME,
    client: Optional[OpenAI] = None,
    backend: str = "openai",
    local_model: Optional[str] = None,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = 120,
) -> Dict:
    """
    Bir QA kaydını seçilen backend ile değerlendir.

    backend:
      - "openai": OpenAI API ile değerlendir (varsayılan).
      - "ollama": Yerel Ollama benzeri HTTP API ile değerlendir.
    """
    backend = (backend or "openai").lower()

    if backend == "ollama":
        model_name = local_model or QA_OLLAMA_MODEL
        return _evaluate_answer_local(
            record=record,
            model=model_name,
            base_url=base_url,
            timeout=timeout,
        )

    # Varsayılan: mevcut OpenAI tabanlı evaluate_answer fonksiyonunu kullan
    return evaluate_answer(
        record=record,
        eval_model=eval_model,
        client=client,
    )


def run_full_pipeline(
    csv_path: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    qdrant_url: str = QDRANT_URL,
    eval_model: str = EVAL_MODEL_NAME,
    k: int = 5,
    openai_client: Optional[OpenAI] = None,
    eval_backend: str = "openai",
    eval_local_model: Optional[str] = None,
    qa_model: str = QA_OLLAMA_MODEL,
    rag_mode: str = "rag",
    eval_enabled: bool = True,
    question_col: str = "question",
    answer_col: str = "answer",
    embed_model: Optional[str] = None,
    think: bool = False,
) -> List[Dict]:
    """
    High-level helper:
      - Load questions from CSV.
      - For each question, retrieve context from Qdrant (via Ollama embeddings).
      - Ask Ollama for an answer.
      - Evaluate the answer and return flat dict rows.

    rag_mode: "rag" | "no_rag" | "both"
    """
    questions = load_questions(csv_path, question_col=question_col, answer_col=answer_col)
    if not questions:
        return []

    eval_backend = (eval_backend or "openai").lower()
    if eval_enabled and eval_backend == "openai" and openai_client is None:
        openai_client = get_openai_client()

    _REMOVE_COLS = {"observation_idea", "ai_strengths", "ai_issues", "ai_suggested_fix"}

    rows: List[Dict] = []

    warmup_model(model=qa_model)

    for item in questions:
        question = item["question"]
        observation_idea = item.get("observation_idea", "")

        # --- RAG'li ---
        if rag_mode in ("rag", "both"):
            chunks = retrieve_chunks(
                question=question,
                collection_name=collection_name,
                k=k,
                qdrant_url=qdrant_url,
                embed_model=embed_model,
            )
            context = "\n\n".join(c["text"] for c in chunks)
            rag_result = generate_rag_answer_ollama(
                question=question,
                context=context,
                model=qa_model,
                think=think,
            )
            record = {
                "model": qa_model,
                "question_index": item["question_index"],
                "question": question,
                "observation_idea": observation_idea,
                "model_answer": rag_result.get("answer", ""),
                "response_time_seconds": rag_result.get("response_time_seconds", 0.0),
                "eval_duration_seconds": rag_result.get("eval_duration_seconds"),
                "tokens_per_second": rag_result.get("tokens_per_second"),
            }
            if eval_enabled:
                eval_fields = evaluate_answer_any(
                    record=record,
                    eval_model=eval_model,
                    client=openai_client if eval_backend == "openai" else None,
                    backend=eval_backend,
                    local_model=eval_local_model,
                )
                eval_row = {**record, **eval_fields}
            else:
                eval_row = {**record}
            for col in _REMOVE_COLS:
                eval_row.pop(col, None)
            eval_row["answer"] = observation_idea
            eval_row["rag_type"] = "RAG'li"
            eval_row["retrieved_chunk_size"] = len(chunks)
            source_parts = []
            for c in chunks:
                src = os.path.basename(c.get("source", "")) or c.get("source", "")
                page = c.get("page", "")
                if src:
                    source_parts.append(f"{src} - page {page}" if page != "" else src)
            eval_row["chunk_sources"] = " | ".join(dict.fromkeys(source_parts))
            eval_row["retrieved_chunks"] = json.dumps([c["text"] for c in chunks], ensure_ascii=False)
            rows.append(eval_row)

        # --- RAG'siz ---
        if rag_mode in ("no_rag", "both"):
            no_rag_result = generate_no_rag_answer_ollama(
                question=question,
                model=qa_model,
                think=think,
            )
            record = {
                "model": qa_model,
                "question_index": item["question_index"],
                "question": question,
                "observation_idea": observation_idea,
                "model_answer": no_rag_result.get("answer", ""),
                "response_time_seconds": no_rag_result.get("response_time_seconds", 0.0),
                "eval_duration_seconds": no_rag_result.get("eval_duration_seconds"),
                "tokens_per_second": no_rag_result.get("tokens_per_second"),
            }
            if eval_enabled:
                eval_fields = evaluate_answer_any(
                    record=record,
                    eval_model=eval_model,
                    client=openai_client if eval_backend == "openai" else None,
                    backend=eval_backend,
                    local_model=eval_local_model,
                )
                eval_row = {**record, **eval_fields}
            else:
                eval_row = {**record}
            for col in _REMOVE_COLS:
                eval_row.pop(col, None)
            eval_row["answer"] = observation_idea
            eval_row["rag_type"] = "RAG'siz"
            eval_row["retrieved_chunk_size"] = 0
            eval_row["chunk_sources"] = ""
            eval_row["retrieved_chunks"] = ""
            rows.append(eval_row)

    return rows


def write_results_to_csv(
    rows: Sequence[Dict],
    output_path: Union[str, TextIO],
    delimiter: str = ";",
) -> int:
    """
    Write evaluation rows to a CSV file (one row per dict).
    Returns number of rows written.
    """
    if not rows:
        if isinstance(output_path, str):
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                f.write("")
        else:
            output_path.write("")
        return 0

    fieldnames = list(rows[0].keys())

    if isinstance(output_path, str):
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    else:
        writer = csv.DictWriter(output_path, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return len(rows)


if __name__ == "__main__":
    # Simple CLI usage example:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run RAG + evaluation pipeline over a CSV file."
    )
    parser.add_argument("csv_path", help="Input CSV path (e.g. sample_rag_input.csv)")
    parser.add_argument(
        "--out",
        dest="output_path",
        default="output.csv",
        help="Output CSV path (default: output.csv)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of context chunks to retrieve from Chroma.",
    )
    args = parser.parse_args()

    client = get_openai_client()
    rows_ = run_full_pipeline(
        csv_path=args.csv_path,
        k=args.k,
        openai_client=client,
        qdrant_url=QDRANT_URL,
    )
    written = write_results_to_csv(rows_, args.output_path)
    print(f"Wrote {written} rows to {args.output_path}")
