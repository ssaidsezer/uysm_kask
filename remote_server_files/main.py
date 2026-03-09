from __future__ import annotations

import asyncio
import io
import os
import re
from typing import Any, List, Tuple

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(title="Remote TTS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Endpoint'ler
# -----------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# TTS – HuggingFace VITS modelleri (model adı istekle belirlenir)
# ---------------------------------------------------------------------------

DEFAULT_TTS_MODEL: str = os.getenv("TTS_MODEL", "facebook/mms-tts-tur")

# Model cache: {model_name: (model, tokenizer)}
_tts_cache: dict[str, Tuple[Any, Any]] = {}


def _get_tts_model(model_name: str) -> Tuple[Any, Any]:
    if model_name in _tts_cache:
        return _tts_cache[model_name]
    from transformers import VitsModel, AutoTokenizer

    model = VitsModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _tts_cache[model_name] = (model, tokenizer)
    return model, tokenizer


def _synthesize_bytes(text: str, model_name: str) -> bytes:
    from scipy.io.wavfile import write as wav_write  # type: ignore[import-untyped]

    model, tokenizer = _get_tts_model(model_name)
    sample_rate: int = model.config.sampling_rate

    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    chunks = [p.strip() for p in parts if p.strip()] if len(text) > 200 else [text]

    waveforms: List[np.ndarray] = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
        waveforms.append(out.waveform.squeeze().cpu().numpy())

    full = np.concatenate(waveforms) if len(waveforms) > 1 else waveforms[0]
    peak = np.max(np.abs(full))
    if peak > 0:
        full = full / peak
    int16 = (full * 32767).astype(np.int16)

    buf = io.BytesIO()
    wav_write(buf, sample_rate, int16)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# TTS API modeli
# ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    text: str
    model: str = DEFAULT_TTS_MODEL


# ---------------------------------------------------------------------------
# TTS Endpoint
# ---------------------------------------------------------------------------

@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """Metni belirtilen VITS modeliyle sese çevirir; WAV döndürür."""
    loop = asyncio.get_running_loop()
    wav_bytes = await loop.run_in_executor(None, _synthesize_bytes, req.text, req.model)
    return Response(content=wav_bytes, media_type="audio/wav")


