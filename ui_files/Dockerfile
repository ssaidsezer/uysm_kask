FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Bazı Python paketleri için gerekli temel sistem paketleri
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# UI tarafında gereken proje dosyaları
COPY streamlit_app.py .
COPY pipeline.py .
COPY rag_index.py .
COPY voice_utils.py .
COPY sample_rag_input.csv .
COPY readme.txt .

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
