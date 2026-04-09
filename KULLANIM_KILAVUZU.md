# UYSM KASK – Kullanım Kılavuzu

## Genel Bakış

Bu proje, askeri eğitim dokümanlarını (PDF) indeksleyip Türkçe soru-cevap yapabilen, cevapları otomatik değerlendiren ve isteğe bağlı olarak seslendiren bir **RAG (Retrieval-Augmented Generation)** sistemidir.

### Sistem Bileşenleri

```
┌─────────────────────────────────────────────────────────┐
│              Web istemcisi (Vite + React)               │
│              http://localhost:5173                       │
│   └── web_frontend  →  FastAPI REST API                 │
└───────────────────────────┬─────────────────────────────┘
                            │  HTTP
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
  Ollama (:11434)    Qdrant (:6333)     TTS Sunucusu (:8000)
  (LLM + Embed)      (Vektör DB)        (FastAPI + HuggingFace)
```

Sunucu tarafı `pipeline.py` ve `rag_index.py` üzerinden PDF indeksleme, retrieval ve değerlendirme mantığını çalıştırır; `voice_utils.py` uzak TTS API’sine HTTP ile bağlanır.

| Bileşen | Görev | Varsayılan Adres |
|---------|-------|-----------------|
| Web arayüzü (Vite) | React SPA | `http://localhost:5173` |
| FastAPI backend | REST API | `http://localhost:8000` (örnek) |
| Ollama | Embedding + Soru-cevap LLM'i | `http://192.168.1.151:11434` |
| Qdrant | Vektör veritabanı | `http://192.168.0.149:6333` |
| TTS API | Metin-sese dönüştürme | `http://192.168.1.151:8000` |

---

## 1. Ortam Kurulumu

### 1.1 Python ve API

Proje kökünden:

```bash
pip install -r requirements.txt
```

Backend’i çalıştırma (örnek):

```bash
python -m uvicorn web_backend.main:app --reload --host 0.0.0.0 --port 8000
```

Sağlık kontrolü:

```bash
python scripts/verify_web_api.py http://127.0.0.1:8000
```

### 1.2 Web arayüzü

```bash
cd web_frontend/
npm install
npm run dev
```

Tarayıcıda `http://localhost:5173` adresini açın. API adresi genelde `http://127.0.0.1:8000` olacak şekilde yapılandırılır (geliştirme ortamında Vite proxy veya ortam değişkenleri kullanılabilir).

### 1.3 Gerekli Modeller (Ollama)

Ollama sunucusunda şu modellerin kurulu olması gerekir:

```bash
# Embedding modeli
ollama pull nomic-embed-text

# Soru-cevap modeli (varsayılan)
ollama pull qwen3:1.7b
```

---

## 2. TTS (Seslendirme) Sunucusunu Başlatma

TTS sunucusu ayrı bir bilgisayarda (veya aynı makinede) Docker ile çalışır.

```bash
cd remote_server_files/
docker compose up -d --build
```

Sağlık kontrolü:

```bash
curl http://<TTS_SUNUCU_IP>:8000/health
# Beklenen: {"status": "ok"}
```

Mevcut TTS modellerini listele:

```bash
curl http://<TTS_SUNUCU_IP>:8000/models
```

**Varsayılan TTS modeli:** `facebook/mms-tts-tur` (Türkçe)

Desteklenen diğer modeller:
- `microsoft/speecht5_tts` (İngilizce, `speaker_id` parametresi alır)
- HuggingFace'te bulunan herhangi bir `text-to-speech` pipeline modeli

---

## 3. Web Arayüzü Sekmeleri

| Sekme | İçerik |
|-------|--------|
| PDF İndeksleme | PDF yükleme, klasik veya akıllı indeksleme, Qdrant’a yazma |
| CSV Değerlendirme | Toplu soru-cevap CSV ile pipeline çalıştırma ve sonuç indirme |
| Manuel Chat Eval | Tek soru ile RAG / RAG’siz karşılaştırma ve chunk görüntüleme |
| Sesli Değerlendirme | Uzak TTS ile ses üretimi |
| Prompt & Model | Model profilleri ve prompt ayarları |
| Yönetim | Ollama/Qdrant bağlantıları, modeller, koleksiyonlar |
| Sonuç Analizi | Dışa aktarılan CSV ile grafik ve özet metrikler |

### 3.1 CSV ile toplu değerlendirme

1. Sorularınızı uygun formatta bir CSV’ye yazın (uygulama ve API şemasına göre sütun adları kullanın).
2. **CSV Değerlendirme** sekmesinden dosyayı yükleyin ve çalıştırın.
3. RAG modunu (RAG’li, RAG’siz veya ikisi) seçin; QA ve değerlendirme modellerini belirleyin.

Sonuçlar `;` ile ayrılmış CSV olarak indirilebilir. Tipik sütunlar:

| Sütun | Açıklama |
|-------|----------|
| `model` | Kullanılan QA modeli |
| `question` | Soru |
| `model_answer` | Modelin cevabı |
| `response_time_seconds` | Cevap süresi |
| `ai_verdict` | AI değerlendirmesi |
| `ai_score` | Puan (0–10) |
| `ai_hallucination_risk` | Hallüsinasyon riski |
| `rag_type` | RAG’li mi RAG’siz mi |

---

## 4. Proje Dosya Yapısı

```
uysm_kask/
├── pipeline.py               # RAG pipeline ve değerlendirme mantığı
├── rag_index.py              # PDF indeksleme ve Qdrant retrieval
├── voice_utils.py            # TTS API istemcisi
├── sample_rag_input.csv      # Örnek soru-cevap CSV (API örnekleri)
├── .env                      # Ortam değişkenleri
│
├── web_backend/              # FastAPI uygulaması
│   ├── main.py
│   ├── requirements.txt
│   └── ...
│
├── web_frontend/             # Vite + React arayüzü
│   └── ...
│
├── remote_server_files/
│   ├── main.py               # FastAPI TTS sunucusu
│   ├── Dockerfile
│   ├── docker-compose.yaml
│   └── requirements-server.txt
│
└── tmp/
    ├── sample_rag_input.csv
    ├── askeri_egitim_kitabi.pdf
    └── taktik_muharebe_yarali_bakimi_el_kitabi.pdf
```

---

## 5. Sorun Giderme

### Ollama'ya bağlanılamıyor
- `.env` dosyasındaki `OLLAMA_HOST` ve `OLLAMA_BASE_URL` adreslerini kontrol edin.
- Ollama servisinin çalıştığını doğrulayın: `curl http://<IP>:11434/api/tags`
- Güvenlik duvarında `11434` portuna izin verildiğinden emin olun.

### Qdrant bağlantı hatası
- `QDRANT_URL` değişkenini kontrol edin.
- `curl http://<IP>:6333/collections` ile Qdrant'ın çalıştığını doğrulayın.

### TTS çalışmıyor
- `VOICE_API_URL` değişkenini kontrol edin.
- `curl http://<IP>:8000/health` ile TTS sunucusunun sağlıklı olduğunu doğrulayın.
- TTS sunucusunda modelin indirilmiş olduğunu `curl http://<IP>:8000/models` ile kontrol edin.
- HuggingFace modelini ilk çalıştırmada sunucu otomatik indirir; bu işlem internet bağlantısı gerektirir.

### OpenAI değerlendirmesi çalışmıyor
- `OPENAI_API_KEY` değişkeninin `.env` dosyasında doğru tanımlı olduğunu kontrol edin.
- Alternatif olarak değerlendirme backend'ini **Ollama** olarak ayarlayıp yerel model kullanın.

### Embedding hatası
- `OLLAMA_EMBED_MODEL` değişkenindeki modelin Ollama'da kurulu olduğunu kontrol edin.
- `ollama pull nomic-embed-text` komutu ile modeli indirin.

### API yanıt vermiyor
- Backend’in çalıştığını doğrulayın: `python scripts/verify_web_api.py http://127.0.0.1:8000`
- Frontend’in API taban adresinin backend ile uyumlu olduğundan emin olun.

---
