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

# torch zaten base image'da mevcut; sadece diğer bağımlılıkları kur
RUN pip install --no-cache-dir -r requirements-server.txt

COPY main.py .

# HuggingFace model cache'i volume'a yönlendirilmiş olacak
ENV HF_HOME=/app/hf_cache

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
