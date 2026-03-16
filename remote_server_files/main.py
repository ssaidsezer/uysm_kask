from __future__ import annotations

import asyncio
import io
import os
import re
from typing import Any, List, Tuple

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
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

_tts_cache: dict[str, Any] = {}
_speecht5_cache: dict[str, Any] = {}


def _normalize_model_name(model_name: str) -> str:
    # UI'da bazen vocoder modeli seçilebiliyor; bunu ana SpeechT5 modeline yönlendir.
    if model_name == "microsoft/speecht5_hifigan":
        return "microsoft/speecht5_tts"
    return model_name


def _get_device() -> tuple[int, str]:
    if torch.cuda.is_available():
        return 0, "cuda"
    return -1, "cpu"


def _get_tts_pipeline(model_name: str) -> Any:
    if model_name in _tts_cache:
        return _tts_cache[model_name]
    from transformers import pipeline

    device_idx, _ = _get_device()
    kwargs: dict[str, Any] = {"task": "text-to-speech", "model": model_name, "device": device_idx}
    # Qwen / fish gibi bazı modeller custom code gerektirebilir.
    if "qwen" in model_name.lower() or "fish" in model_name.lower() or "mlx-community" in model_name.lower():
        kwargs["trust_remote_code"] = True

    pipe = pipeline(**kwargs)
    _tts_cache[model_name] = pipe
    return pipe


def _get_speecht5_stack() -> tuple[Any, Any, Any, str]:
    cached = _speecht5_cache.get("stack")
    if cached is not None:
        return cached

    from transformers import (  # type: ignore[import-untyped]
        AutoTokenizer,
        SpeechT5ForTextToSpeech,
        SpeechT5HifiGan,
    )

    _, device_name = _get_device()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/speecht5_tts")
    # torch.load güvenlik kısıtına takılmamak için safetensors ağırlıklarını zorla.
    model = SpeechT5ForTextToSpeech.from_pretrained(
        "microsoft/speecht5_tts",
        use_safetensors=True,
    ).to(device_name)
    vocoder = SpeechT5HifiGan.from_pretrained(
        "microsoft/speecht5_hifigan",
        use_safetensors=True,
    ).to(device_name)

    stack = (tokenizer, model, vocoder, device_name)
    _speecht5_cache["stack"] = stack
    return stack


def _speaker_embedding_from_id(speaker_id: str | None, device_name: str) -> torch.Tensor:
    seed_val = 4312
    if speaker_id:
        try:
            seed_val = int(speaker_id)
        except ValueError:
            seed_val = abs(hash(speaker_id)) % (2**31 - 1)

    generator = torch.Generator()
    generator.manual_seed(seed_val)
    return torch.randn((1, 512), generator=generator).to(device_name)


def _synthesize_speecht5(text: str, speaker_id: str | None) -> Tuple[bytes, float]:
    from scipy.io.wavfile import write as wav_write  # type: ignore[import-untyped]

    tokenizer, model, vocoder, device_name = _get_speecht5_stack()
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device_name)
    speaker_embeddings = _speaker_embedding_from_id(speaker_id, device_name)

    with torch.no_grad():
        speech = model.generate_speech(input_ids, speaker_embeddings, vocoder=vocoder)

    if isinstance(speech, torch.Tensor):
        arr = speech.detach().float().cpu().numpy()
    else:
        arr = np.asarray(speech, dtype=np.float32)

    if arr.ndim > 1:
        arr = arr.squeeze()

    sample_rate = 16_000
    duration_sec = float(len(arr) / sample_rate) if len(arr) > 0 else 0.0
    peak = float(np.max(np.abs(arr))) if len(arr) > 0 else 0.0
    if peak > 0:
        arr = arr / peak
    int16 = (arr * 32767).astype(np.int16)

    buf = io.BytesIO()
    wav_write(buf, sample_rate, int16)
    return buf.getvalue(), duration_sec


def _synthesize_with_pipeline(
    text: str,
    model_name: str,
    speaker_id: str | None,
    voice_preset: str | None,
) -> Tuple[bytes, float]:
    from scipy.io.wavfile import write as wav_write  # type: ignore[import-untyped]

    pipe = _get_tts_pipeline(model_name)

    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    chunks = [p.strip() for p in parts if p.strip()] if len(text) > 200 else [text]

    waveforms: List[np.ndarray] = []
    sample_rate = 16000  # Varsayılan (eğer model söylemezse)
    
    for chunk in chunks:
        call_kwargs: dict[str, Any] = {}
        # Bazı custom modeller ekstra alan kabul edebilir; kabul etmeyenlerde sessizce düş.
        if speaker_id:
            call_kwargs["speaker_id"] = speaker_id
        if voice_preset and voice_preset != "Varsayılan":
            call_kwargs["voice_preset"] = voice_preset

        try:
            out = pipe(chunk, **call_kwargs)
        except TypeError:
            out = pipe(chunk)

        if isinstance(out, dict):
            if "sampling_rate" in out:
                sample_rate = out["sampling_rate"]
            audio_chunk = out["audio"]
        else:
            # Bazı custom pipeline'lar doğrudan ndarray döndürebilir.
            audio_chunk = out
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


def _synthesize_bytes(
    text: str,
    model_name: str,
    speaker_id: str | None = None,
    voice_preset: str | None = None,
) -> Tuple[bytes, float]:
    resolved_model = _normalize_model_name(model_name)

    # SpeechT5 ayrı inference yoluna ihtiyaç duyuyor.
    if resolved_model == "microsoft/speecht5_tts":
        return _synthesize_speecht5(text, speaker_id)

    try:
        return _synthesize_with_pipeline(text, resolved_model, speaker_id, voice_preset)
    except Exception:
        # Son çare: model-specific hata olursa varsayılan modele düş.
        if resolved_model != DEFAULT_TTS_MODEL:
            return _synthesize_with_pipeline(
                text,
                DEFAULT_TTS_MODEL,
                speaker_id=None,
                voice_preset=None,
            )
        raise


# ---------------------------------------------------------------------------
# TTS API modeli
# ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    text: str
    model: str = DEFAULT_TTS_MODEL
    speaker_id: str | None = None
    voice_preset: str | None = None


# ---------------------------------------------------------------------------
# TTS Endpoint
# ---------------------------------------------------------------------------

@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """Metni belirtilen VITS modeliyle sese çevirir; WAV döndürür."""
    loop = asyncio.get_running_loop()
    try:
        wav_bytes, duration_sec = await loop.run_in_executor(
            None,
            _synthesize_bytes,
            req.text,
            req.model,
            req.speaker_id,
            req.voice_preset,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS sentezleme hatası: {exc}") from exc
    return Response(
        content=wav_bytes, 
        media_type="audio/wav",
        headers={"X-Audio-Duration": f"{duration_sec:.2f}"}
    )


