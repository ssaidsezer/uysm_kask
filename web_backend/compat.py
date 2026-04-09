"""Helpers mirrored from streamlit_app.py without Streamlit (HTTP + pure logic)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Tuple

import requests

from rag_index import DEFAULT_COLLECTION_NAME, QDRANT_URL

WORKSPACE_DIR = Path(__file__).resolve().parent.parent


def get_ollama_base_url() -> str:
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


def collection_name_full(
    base: str, embed_model: str, chunk_size: int, chunk_overlap: int
) -> str:
    safe = embed_model.replace(":", "_").replace("/", "_").replace(".", "_")
    return f"{base}_{safe}_{chunk_size}c_{chunk_overlap}ov"


def smart_collection_name_full(
    base: str,
    embed_model: str,
    parent_size: int,
    child_size: int,
    child_overlap: int,
) -> str:
    safe = embed_model.replace(":", "_").replace("/", "_").replace(".", "_")
    return f"{base}_{safe}_{parent_size}p_{child_size}c_{child_overlap}ov"


def collection_options_for_embed(
    collections: List[str],
    embed_model: str | None,
) -> tuple[List[str], List[str]]:
    if not embed_model:
        return [], []
    safe_embed = embed_model.replace(":", "_").replace("/", "_").replace(".", "_")
    classic_cols = sorted(
        c
        for c in collections
        if safe_embed in c and not c.endswith("_children") and not c.endswith("_parents")
    )
    smart_bases = sorted(
        {
            c[: -len("_children")]
            for c in collections
            if c.endswith("_children")
            and safe_embed in c
            and f"{c[: -len('_children')]}_parents" in collections
        }
    )
    return classic_cols, smart_bases


def is_embedding_model(host: str, model_name: str) -> bool:
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

    capabilities = data.get("capabilities") or []
    if capabilities:
        return "embedding" in capabilities

    name_lower = model_name.lower()
    if any(kw in name_lower for kw in ("embed", "bge", "e5", "gte", "rerank")):
        return True

    template = (data.get("template") or "").strip()
    if not template:
        return True

    return False


def list_ollama_models() -> tuple[List[str], str, float, int]:
    host = get_ollama_base_url()
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
        return [], (
            f"Uzak Ollama sunucusuna ({host}) bağlanılamadı. "
            "Lütfen sunucunun açık olduğundan emin olun."
        ), 0.0, 0

    all_names: List[str] = []
    for item in data.get("models", []):
        name = item.get("name")
        if isinstance(name, str):
            all_names.append(name)

    t0 = time.time()
    models: List[str] = []
    filtered_count = 0
    for name in all_names:
        if is_embedding_model(host, name):
            filtered_count += 1
        else:
            models.append(name)
    elapsed = round(time.time() - t0, 2)

    return models, "", elapsed, filtered_count


def list_embedding_models() -> tuple[List[str], str]:
    host = get_ollama_base_url()
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

    embed_models = [name for name in all_names if is_embedding_model(host, name)]
    return embed_models, ""


def ensure_tmp_dir() -> Path:
    tmp_dir = WORKSPACE_DIR / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    return tmp_dir


def pull_ollama_model(model_name: str) -> tuple[bool, str]:
    host = get_ollama_base_url()
    if not host:
        return False, "OLLAMA_BASE_URL veya OLLAMA_HOST ortam değişkeni tanımlı değil."
    url = host + "/api/pull"
    try:
        resp = requests.post(url, json={"name": model_name, "stream": False}, timeout=300)
        resp.raise_for_status()
        return True, f"'{model_name}' başarıyla pull edildi."
    except Exception as e:
        return False, f"Pull sırasında hata: {e}"


def delete_ollama_model(model_name: str) -> tuple[bool, str]:
    host = get_ollama_base_url()
    if not host:
        return False, "OLLAMA_BASE_URL veya OLLAMA_HOST ortam değişkeni tanımlı değil."
    url = host + "/api/delete"
    try:
        resp = requests.delete(url, json={"name": model_name}, timeout=30)
        resp.raise_for_status()
        return True, f"'{model_name}' başarıyla silindi."
    except Exception as e:
        return False, f"Silme sırasında hata: {e}"


def list_qdrant_collections() -> tuple[List[str], str]:
    qdrant_url = os.environ.get("QDRANT_URL", QDRANT_URL)
    try:
        resp = requests.get(f"{qdrant_url}/collections", timeout=10)
        resp.raise_for_status()
        data = resp.json() or {}
        collections = [c["name"] for c in data.get("result", {}).get("collections", [])]
        return collections, ""
    except Exception as e:
        return [], f"Qdrant koleksiyonları listelenemedi: {e}"


def delete_qdrant_collection(collection_name: str) -> tuple[bool, str]:
    qdrant_url = os.environ.get("QDRANT_URL", QDRANT_URL)
    try:
        resp = requests.delete(f"{qdrant_url}/collections/{collection_name}", timeout=30)
        resp.raise_for_status()
        return True, f"'{collection_name}' koleksiyonu başarıyla silindi."
    except Exception as e:
        return False, f"Koleksiyon silme sırasında hata: {e}"


def list_all_ollama_model_names_raw() -> tuple[List[str], str]:
    host = get_ollama_base_url()
    if not host:
        return [], "OLLAMA_BASE_URL veya OLLAMA_HOST tanımlı değil."
    try:
        resp = requests.get(host + "/api/tags", timeout=10)
        resp.raise_for_status()
        names = [
            item["name"]
            for item in (resp.json() or {}).get("models", [])
            if isinstance(item.get("name"), str)
        ]
        return names, ""
    except Exception as e:
        return [], str(e)


def get_monitor_stats() -> tuple[dict | None, str]:
    monitor_url = os.environ.get("MONITOR_URL", "http://192.168.1.151:8081").rstrip("/")
    try:
        resp = requests.get(f"{monitor_url}/stats", timeout=3)
        if resp.status_code == 200:
            return resp.json(), ""
        return None, "Sistem bilgisi alınamadı"
    except Exception:
        return None, f"Monitor erişilemiyor ({monitor_url})"
