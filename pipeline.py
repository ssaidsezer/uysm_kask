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
    retrieve_context,
)


QA_OLLAMA_MODEL = os.getenv("QA_OLLAMA_MODEL", "qwen3:1.7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "")
EVAL_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", "gpt-4.1-mini")


def load_questions(csv_path: str) -> List[Dict]:
    """
    Load questions and optional observation_idea (Answers) from a CSV.
    Assumes headers: Questions, Answers (like sample_rag_input.csv).
    """
    items: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            question = (row.get("Questions") or "").strip()
            observation = (row.get("Answers") or "").strip()
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


def generate_rag_answer_ollama(
    question: str,
    context: str,
    model: str = QA_OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = 120,
) -> Dict:
    """
    Call a local Ollama model (e.g. Qwen3 1.7B) with question + context.

    Returns a dict with:
      - answer: str
      - response_time_seconds: float
    """
    prompt = _build_rag_prompt(question, context)
    if not base_url:
        raise ValueError(
            "OLLAMA_BASE_URL ortam değişkeni tanımlı değil. "
            "Lütfen .env dosyasına uzak sunucu adresini ekleyin (örn: OLLAMA_BASE_URL=http://192.168.1.151:11434)."
        )
    t0 = time.time()

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
    dt = time.time() - t0
    resp.raise_for_status()
    data = resp.json()

    answer = ""
    # Typical Ollama /api/chat response: {"message": {"role": "assistant", "content": "..."}}
    if isinstance(data, dict):
        message = data.get("message") or {}
        if isinstance(message, dict):
            answer = str(message.get("content") or "")
        elif "choices" in data:
            # Fallback if future API resembles OpenAI style
            choices = data.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                answer = str(msg.get("content") or "")

    answer = answer.replace("\r\n", "\n").strip()

    # Optional token-level statistics from Ollama
    eval_count = None
    eval_duration_seconds = None
    tokens_per_second = None
    if isinstance(data, dict):
        raw_eval_count = data.get("eval_count")
        raw_eval_duration = data.get("eval_duration")
        if isinstance(raw_eval_count, (int, float)) and isinstance(
            raw_eval_duration, (int, float)
        ):
            eval_count = int(raw_eval_count)
            # Ollama durations are in nanoseconds
            eval_duration_seconds = float(raw_eval_duration) / 1e9
            if eval_duration_seconds > 0:
                tokens_per_second = eval_count / eval_duration_seconds

    return {
        "answer": answer,
        "response_time_seconds": dt,
        "eval_count": eval_count,
        "eval_duration_seconds": eval_duration_seconds,
        "tokens_per_second": tokens_per_second,
    }


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
) -> Dict:
    """
    Call a local Ollama model WITHOUT any retrieved context (no RAG).

    Returns a dict with:
      - answer: str
      - response_time_seconds: float
    """
    prompt = _build_no_rag_prompt(question)
    if not base_url:
        raise ValueError(
            "OLLAMA_BASE_URL ortam değişkeni tanımlı değil. "
            "Lütfen .env dosyasına uzak sunucu adresini ekleyin (örn: OLLAMA_BASE_URL=http://192.168.1.151:11434)."
        )
    t0 = time.time()

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
    dt = time.time() - t0
    resp.raise_for_status()
    data = resp.json()

    answer = ""
    if isinstance(data, dict):
        message = data.get("message") or {}
        if isinstance(message, dict):
            answer = str(message.get("content") or "")
        elif "choices" in data:
            choices = data.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                answer = str(msg.get("content") or "")

    answer = answer.replace("\r\n", "\n").strip()

    # Optional token-level statistics from Ollama
    eval_count = None
    eval_duration_seconds = None
    tokens_per_second = None
    if isinstance(data, dict):
        raw_eval_count = data.get("eval_count")
        raw_eval_duration = data.get("eval_duration")
        if isinstance(raw_eval_count, (int, float)) and isinstance(
            raw_eval_duration, (int, float)
        ):
            eval_count = int(raw_eval_count)
            eval_duration_seconds = float(raw_eval_duration) / 1e9
            if eval_duration_seconds > 0:
                tokens_per_second = eval_count / eval_duration_seconds

    return {
        "answer": answer,
        "response_time_seconds": dt,
        "eval_count": eval_count,
        "eval_duration_seconds": eval_duration_seconds,
        "tokens_per_second": tokens_per_second,
    }


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
        "You are an evaluator in an automated pipeline.\n"
        "Return ONE flat JSON object with exactly these keys:\n"
        "{\n"
        '  "model": string,\n'
        '  "question_index": integer,\n'
        '  "question": string,\n'
        '  "observation_idea": string,\n'
        '  "model_answer": string,\n'
        '  "response_time_seconds": number,\n'
        '  "ai_verdict": string,\n'
        '  "ai_score": integer,\n'
        '  "ai_hallucination_risk": string,\n'
        '  "ai_strengths": string,\n'
        '  "ai_issues": string,\n'
        '  "ai_suggested_fix": string\n'
        "}\n"
        "JSON only, no extra text."
    )

    user = (
        f'model: "{record["model"]}"\n'
        f"question_index: {int(record['question_index'])}\n"
        f'question: "{record["question"]}"\n'
        f'observation_idea: "{record.get("observation_idea", "")}"\n'
        f'model_answer: "{record.get("model_answer", "")}"\n'
        f"response_time_seconds: {float(record.get('response_time_seconds', 0.0))}\n"
    )

    response = client.chat.completions.create(
        model=eval_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
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
        "You are an evaluator in an automated pipeline.\n"
        "Return ONE flat JSON object with exactly these keys:\n"
        "{\n"
        '  \"model\": string,\n'
        '  \"question_index\": integer,\n'
        '  \"question\": string,\n'
        '  \"observation_idea\": string,\n'
        '  \"model_answer\": string,\n'
        '  \"response_time_seconds\": number,\n'
        '  \"ai_verdict\": string,\n'
        '  \"ai_score\": integer,\n'
        '  \"ai_hallucination_risk\": string,\n'
        '  \"ai_strengths\": string,\n'
        '  \"ai_issues\": string,\n'
        '  \"ai_suggested_fix\": string\n'
        "}\n"
        "JSON only, no extra text."
    )

    user = (
        f'model: "{record["model"]}"\n'
        f"question_index: {int(record['question_index'])}\n"
        f'question: "{record["question"]}"\n'
        f'observation_idea: "{record.get("observation_idea", "")}"\n'
        f'model_answer: "{record.get("model_answer", "")}"\n'
        f"response_time_seconds: {float(record.get('response_time_seconds', 0.0))}\n"
    )

    prompt = (
        system
        + "\n\n"
        + user
        + "\n\nYukarıdaki talimatlara göre SADECE geçerli bir JSON nesnesi üret."
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
) -> List[Dict]:
    """
    High-level helper:
      - Load questions from CSV.
      - For each question, retrieve context from Qdrant (via Ollama embeddings).
      - Ask Ollama for an answer.
      - Evaluate the answer and return flat dict rows.
    """
    questions = load_questions(csv_path)
    if not questions:
        return []

    eval_backend = (eval_backend or "openai").lower()
    if eval_backend == "openai" and openai_client is None:
        openai_client = get_openai_client()

    rows: List[Dict] = []

    warmup_model(model=qa_model)

    for item in questions:
        question = item["question"]
        observation_idea = item.get("observation_idea", "")

        context = retrieve_context(
            question=question,
            collection_name=collection_name,
            k=k,
            qdrant_url=qdrant_url,
        )

        rag_result = generate_rag_answer_ollama(
            question=question,
            context=context,
            model=qa_model,
        )

        record = {
            "model": qa_model,
            "question_index": item["question_index"],
            "question": question,
            "observation_idea": observation_idea,
            "model_answer": rag_result.get("answer", ""),
            "response_time_seconds": rag_result.get("response_time_seconds", 0.0),
        }

        eval_row = evaluate_answer_any(
            record=record,
            eval_model=eval_model,
            client=openai_client if eval_backend == "openai" else None,
            backend=eval_backend,
            local_model=eval_local_model,
        )
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

