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

@app.get("/models")
def list_models() -> dict:
    import pathlib
    hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
    hub_path = pathlib.Path(hf_home)
    
    # If the user is running it locally without HF_HOME explicitly, or locally
    # the hub is actually a subdirectory of HF_HOME called "hub"
    if hub_path.name != "hub" and (hub_path / "hub").exists():
        hub_path = hub_path / "hub"
    elif not hub_path.name.endswith("hub") and not (hub_path / "hub").exists():
        # Typically hf_home contains the hub directly or as a subfolder
        # Let's just try hub_path
        pass

    models = []
    if hub_path.exists():
        for d in hub_path.iterdir():
            if d.is_dir() and d.name.startswith("models--"):
                # Convert 'models--facebook--mms-tts-tur' to 'facebook/mms-tts-tur'
                name_parts = d.name.replace("models--", "").split("--", 1)
                models.append("/".join(name_parts))
    return {"downloaded_models": sorted(models)}


# ---------------------------------------------------------------------------
# TTS – HuggingFace pipeline (Bütün modellerle uyumlu olması için pipeline kullanıyoruz)
# İstediğiniz modeli girdiğinizde, eğer yoksa HF otomatik olarak "pull" edecektir.
# ---------------------------------------------------------------------------

DEFAULT_TTS_MODEL: str = os.getenv("TTS_MODEL", "facebook/mms-tts-tur")

# Model cache: {model_name: pipeline}
_tts_cache: dict[str, Any] = {}


def _get_tts_pipeline(model_name: str) -> Any:
    if model_name in _tts_cache:
        return _tts_cache[model_name]
    from transformers import pipeline
    
    # GPU varsa device=0 üzerinden çalıştır (hızlandırır)
    device = 0 if torch.cuda.is_available() else -1
    
    # Hugging Face pipeline'ı model mimarisine (VITS, SpeechT5 vs) bakmaksızın genel bir köprü kurar.
    # Model yüklü değilse, otomatik olarak internetten bulup indirecektir (pull).
    pipe = pipeline("text-to-speech", model=model_name, device=device)
    _tts_cache[model_name] = pipe
    return pipe


def _synthesize_bytes(text: str, model_name: str) -> Tuple[bytes, float]:
    from scipy.io.wavfile import write as wav_write  # type: ignore[import-untyped]

    pipe = _get_tts_pipeline(model_name)

    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    chunks = [p.strip() for p in parts if p.strip()] if len(text) > 200 else [text]

    waveforms: List[np.ndarray] = []
    sample_rate = 16000  # Varsayılan (eğer model söylemezse)
    
    for chunk in chunks:
        # Pipeline çıktısı genelde {'audio': np.ndarray, 'sampling_rate': int} şeklindedir
        out = pipe(chunk)
        if "sampling_rate" in out:
            sample_rate = out["sampling_rate"]
        
        audio_chunk = out["audio"]
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.squeeze()
            
        waveforms.append(audio_chunk)

    full = np.concatenate(waveforms) if len(waveforms) > 1 else waveforms[0]
    
    # Sesin kaç saniye sürdüğünü hesapla
    duration_sec = len(full) / sample_rate

    peak = np.max(np.abs(full))
    if peak > 0:
        full = full / peak
    int16 = (full * 32767).astype(np.int16)

    buf = io.BytesIO()
    wav_write(buf, sample_rate, int16)
    return buf.getvalue(), duration_sec


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
    wav_bytes, duration_sec = await loop.run_in_executor(None, _synthesize_bytes, req.text, req.model)
    return Response(
        content=wav_bytes, 
        media_type="audio/wav",
        headers={"X-Audio-Duration": f"{duration_sec:.2f}"}
    )


