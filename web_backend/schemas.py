"""API request/response models (frozen contract for React client)."""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class AppConfigResponse(BaseModel):
    qdrant_url: str
    default_collection_name: str
    ollama_base_url: str
    monitor_url: str
    qa_ollama_model: str
    eval_model_name: str
    chunk_size_default: int
    chunk_overlap_default: int
    smart_parent_size: int
    smart_child_size: int
    smart_child_overlap: int
    smart_boundary_llm_model: str


class MonitorStats(BaseModel):
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    vram_used: Optional[float] = None
    vram_total: Optional[float] = None
    ram_used: Optional[float] = None
    ram_total: Optional[float] = None
    storage_used: Optional[float] = None
    storage_total: Optional[float] = None


class ConnectionStatusResponse(BaseModel):
    qdrant_ok: bool
    qdrant_message: str
    ollama_ok: bool
    ollama_message: str
    monitor: Optional[MonitorStats] = None
    monitor_error: Optional[str] = None


class OllamaModelsResponse(BaseModel):
    models: List[str]
    error: str
    filter_elapsed_seconds: float
    filtered_embedding_count: int


class EmbeddingModelsResponse(BaseModel):
    models: List[str]
    error: str


class CollectionsResponse(BaseModel):
    collections: List[str]
    error: str


class PullModelRequest(BaseModel):
    name: str


class DeleteModelRequest(BaseModel):
    name: str


class SimpleMessageResponse(BaseModel):
    success: bool
    message: str


class JobProgress(BaseModel):
    phase: str
    current: int
    total: int
    elapsed_sec: float
    message: Optional[str] = None


class JobStatusResponse(BaseModel):
    id: str
    status: Literal["pending", "running", "completed", "failed"]
    progress: Optional[JobProgress] = None
    result: Optional[Any] = None
    error: Optional[str] = None


class IndexJobStartResponse(BaseModel):
    job_id: str


class ChatEvalRequest(BaseModel):
    question: str
    expected_answer: str = ""
    rag_mode: Literal["rag", "no_rag", "both"]
    k: int = 5
    qa_models_selected: List[str] = Field(default_factory=list)
    eval_enabled: bool = True
    eval_backend: Literal["OpenAI", "Yerel (Ollama)"] = "OpenAI"
    eval_model_name: str = ""
    local_eval_model_name: Optional[str] = None
    openai_api_key: Optional[str] = None
    collection_name: str = ""
    embed_model: Optional[str] = None
    think: bool = False
    smart_rag: bool = False
    score_threshold: float = 0.55
    retrieval_mode: Literal["vector", "bm25"] = "vector"


class ChunkCard(BaseModel):
    text: str
    score: float
    child_text: Optional[str] = None


class ChatModelResult(BaseModel):
    model_name: str
    error: Optional[str] = None
    rag_answer: Optional[str] = None
    rag_eval: Optional[dict] = None
    rag_result_meta: Optional[dict] = None
    no_rag_answer: Optional[str] = None
    no_rag_eval: Optional[dict] = None
    no_rag_result_meta: Optional[dict] = None


class ChatEvalResponse(BaseModel):
    chunks: List[ChunkCard] = Field(default_factory=list)
    chunk_section_title: str = ""
    chunk_warning: Optional[str] = None
    model_results: List[ChatModelResult] = Field(default_factory=list)
    chat_eval_rows: List[dict] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class TTSRequest(BaseModel):
    text: str
    model: str = "facebook/mms-tts-tur"
    speaker_id: Optional[str] = None
    voice_preset: Optional[str] = None


class TTSModelsResponse(BaseModel):
    downloaded_models: List[str]


class AnalysisResponse(BaseModel):
    warnings: List[str] = Field(default_factory=list)
    info_messages: List[str] = Field(default_factory=list)
    global_summary: Optional[dict] = None
    breakdown: List[dict] = Field(default_factory=list)
    charts: List[dict] = Field(default_factory=list)
    lift_rows: List[dict] = Field(default_factory=list)
    distribution_chart: Optional[dict] = None
    problems_rag_worse: List[dict] = Field(default_factory=list)
    problems_both_low: List[dict] = Field(default_factory=list)
    detail_rows: List[dict] = Field(default_factory=list)
    rag_label: Optional[str] = None
    no_rag_label: Optional[str] = None
