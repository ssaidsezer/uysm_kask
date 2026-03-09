"""STT (Speech-to-Text) and TTS (Text-to-Speech) utilities for the voice tab."""

from __future__ import annotations

import io
import re
from typing import Any, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# STT – openai/whisper-large-v3
# ---------------------------------------------------------------------------

_stt_pipeline = None


def load_stt_model():
    """Lazy-load and cache the Whisper ASR pipeline."""
    global _stt_pipeline
    if _stt_pipeline is not None:
        return _stt_pipeline

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    _stt_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    return _stt_pipeline


def transcribe_audio(audio_bytes: bytes) -> str:
    """Decode *audio_bytes* to 16 kHz mono and transcribe via Whisper."""
    import librosa  # type: ignore[import-untyped]

    pipe = load_stt_model()

    audio_array, _ = librosa.load(io.BytesIO(audio_bytes), sr=16_000, mono=True)
    result: Any = pipe(
        audio_array,
        generate_kwargs={"language": "turkish"},
    )
    return result["text"]


# ---------------------------------------------------------------------------
# TTS – facebook/mms-tts-tur
# ---------------------------------------------------------------------------

_tts_model: Any = None
_tts_tokenizer: Any = None


def load_tts_model():
    """Lazy-load and cache the VITS TTS model + tokenizer."""
    global _tts_model, _tts_tokenizer
    if _tts_model is not None:
        return _tts_model, _tts_tokenizer

    from transformers import VitsModel, AutoTokenizer

    _tts_model = VitsModel.from_pretrained("facebook/mms-tts-tur")
    _tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tur")
    return _tts_model, _tts_tokenizer


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-like chunks for VITS (which struggles with long input)."""
    parts = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def synthesize_speech(text: str) -> Tuple[bytes, int]:
    """Synthesize *text* into a WAV byte buffer. Returns (wav_bytes, sample_rate)."""
    from scipy.io.wavfile import write as wav_write

    model, tokenizer = load_tts_model()
    sample_rate = model.config.sampling_rate

    chunks = _split_sentences(text) if len(text) > 200 else [text]
    waveforms: list[np.ndarray] = []

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)
        waveforms.append(output.waveform.squeeze().cpu().numpy())

    full_waveform = np.concatenate(waveforms) if len(waveforms) > 1 else waveforms[0]

    # Normalise to int16 range for WAV
    peak = np.max(np.abs(full_waveform))
    if peak > 0:
        full_waveform = full_waveform / peak
    int16_wave = (full_waveform * 32767).astype(np.int16)

    buf = io.BytesIO()
    wav_write(buf, sample_rate, int16_wave)
    return buf.getvalue(), sample_rate
