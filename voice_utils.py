"""STT ve TTS için uzak FastAPI endpoint'lerine HTTP istemcisi.

Her iki model de uzak sunucuda (Docker container) çalışır.
Bu modül sadece ses byte'larını /stt'ye, metni /tts'ye POST eder.

Yapılandırma:
    VOICE_API_URL  — FastAPI sunucusunun adresi (varsayılan: http://localhost:8000)
"""

from __future__ import annotations

import os
from typing import Tuple

import requests

VOICE_API_URL: str = os.getenv("VOICE_API_URL", "http://localhost:8000")
TTS_MODEL: str = os.getenv("TTS_MODEL", "facebook/mms-tts-tur")

# MMS-TTS-TUR her zaman 16 kHz çıktı üretir
_TTS_SAMPLE_RATE = 16_000


def synthesize_speech(text: str, model: str = TTS_MODEL) -> Tuple[bytes, int]:
    """Metni uzak VITS modeliyle WAV'a dönüştür."""
    url = f"{VOICE_API_URL.rstrip('/')}/tts"
    resp = requests.post(url, json={"text": text, "model": model}, timeout=180)
    resp.raise_for_status()
    return resp.content, _TTS_SAMPLE_RATE

