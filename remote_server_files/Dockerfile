# ---- Remote RAG + Voice API (GPU) ----
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# libsndfile1: librosa için ses dosyası okuma arka ucu
# ffmpeg: mp3/m4a/ogg gibi formatların çözümlenmesi
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-server.txt .

# Conda'daki torch'u miras alan bir venv oluştur (PEP 668 sorununu temiz şekilde çözer)
RUN python -m venv --system-site-packages /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Ek paketleri venv içine kur (torch zaten conda'dan miras alınıyor)
RUN pip install --no-cache-dir -r requirements-server.txt \
    && python -c "import torch; print('torch OK:', torch.__version__)"

COPY main.py .

# HuggingFace model cache'i volume'a yönlendirilmiş olacak
ENV HF_HOME=/app/hf_cache

# Venv içerisindeki uvicorn'u kullanarak uygulamayı başlat
CMD ["/app/venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
