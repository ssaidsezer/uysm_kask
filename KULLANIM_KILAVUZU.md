# UYSM KASK – Kullanım Kılavuzu

## Genel Bakış

Bu proje, askeri eğitim dokümanlarını (PDF) indeksleyip Türkçe soru-cevap yapabilen, cevapları otomatik değerlendiren ve isteğe bağlı olarak seslendiren bir **RAG (Retrieval-Augmented Generation)** sistemidir.

### Sistem Bileşenleri

```
┌─────────────────────────────────────────────────────────┐
│                     UI Bilgisayarı                      │
│                                                         │
│   Streamlit Arayüzü (:8501)                             │
│   └── rag_index.py   → PDF indeksleme & retrieval       │
│   └── pipeline.py    → Soru-cevap & değerlendirme       │
│   └── voice_utils.py → TTS istemcisi                    │
└───────────────────┬─────────────────────────────────────┘
                    │  HTTP
        ┌───────────┼───────────────────┐
        ▼           ▼                   ▼
  Ollama (:11434) Qdrant (:6333)  TTS Sunucusu (:8000)
  (LLM + Embed)  (Vektör DB)     (FastAPI + HuggingFace)
```

| Bileşen | Görev | Varsayılan Adres |
|---------|-------|-----------------|
| Streamlit UI | Web arayüzü | `http://localhost:8501` |
| Ollama | Embedding + Soru-cevap LLM'i | `http://192.168.1.151:11434` |
| Qdrant | Vektör veritabanı | `http://192.168.0.149:6333` |
| TTS API | Metin-sese dönüştürme | `http://192.168.1.151:8000` |

---

## 1. Ortam Kurulumu

### 1.1 Gerekli Modeller

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

## 3. Streamlit Arayüzünü Başlatma

### Yerel Geliştirme

```bash
pip install -r ui_files/requirements-ui.txt
streamlit run streamlit_app.py
```

### Docker ile

```bash
cd ui_files/
docker compose up -d --build
```

Arayüz: `http://localhost:8501`

---

## 4. Arayüzün Sekmeleri ve Kullanımı

### 4.1 PDF Indeksleme

1. Arayüzde **"PDF & İndeksleme"** bölümüne gidin.
2. Bir veya birden fazla PDF dosyası yükleyin.
3. **"İndeksle"** butonuna tıklayın.

Süreç üç aşamada ilerlenir:
```
PDF Metni Çıkar → Ollama Embedding → Qdrant'a Kaydet
```

### 4.2 Soru-Cevap (RAG Değerlendirme)

1. **CSV Dosyası Hazırlama:** Sorularınızı aşağıdaki formatta bir CSV'ye yazın:

   ```csv
   Questions,Answers
   "İlk yardım nedir?","Hasta ve yaralıya yapılan acil bakımdır."
   "Kalp durmasının belirtileri nedir?","Nabız ve nefes durur, göz bebekleri büyür."
   ```

   > `Answers` sütunu opsiyoneldir; doldurulursa değerlendirme referansı olarak kullanılır.

2. CSV'yi arayüze yükleyin.
3. Çalıştırma modunu seçin:
   - **RAG'li:** Soruları Qdrant'tan alınan bağlamla cevaplar
   - **RAG'siz:** Soruları sadece modelin genel bilgisiyle cevaplar
   - **Her İkisi:** İki modu karşılaştırmalı çalıştırır

4. QA modelini ve değerlendirme modelini seçin.
5. **"Çalıştır"** butonuna tıklayın.

Sonuçlar `;` ile ayrılmış CSV olarak indirilebilir. Sütunlar:

| Sütun | Açıklama |
|-------|----------|
| `model` | Kullanılan QA modeli |
| `question` | Soru |
| `model_answer` | Modelin cevabı |
| `response_time_seconds` | Cevap süresi |
| `ai_verdict` | AI değerlendirmesi (DOĞRU/YANLIŞ/KISMEN vb.) |
| `ai_score` | Puan (0–10) |
| `ai_hallucination_risk` | Hallüsinasyon riski (DÜŞÜK/ORTA/YÜKSEK) |
| `ai_strengths` | Cevabın güçlü yönleri |
| `ai_issues` | Cevabın zayıf yönleri |
| `ai_suggested_fix` | Önerilen düzeltme |
| `retrieved_chunks` | Qdrant'tan alınan bağlam metni |
| `rag_type` | RAG'li mi RAG'siz mi |

### 4.3 TTS (Metin-Ses Dönüştürme)

1. **"TTS"** sekmesine gidin.
2. Seslendirilecek metni girin.
3. TTS modelini seçin (sunucudan mevcut modeller çekilir).
4. **"Seslendir"** butonuna tıklayın.
5. Üretilen WAV dosyası arayüzde oynatılabilir.

---

## 5. Proje Dosya Yapısı

```
uysm_kask/
├── streamlit_app.py          # Streamlit web arayüzü
├── pipeline.py               # RAG pipeline ve değerlendirme mantığı
├── rag_index.py              # PDF indeksleme ve Qdrant retrieval
├── voice_utils.py            # TTS API istemcisi
├── .env                      # Ortam değişkenleri (her makinede ayrı ayarlanır)
│
├── ui_files/
│   ├── Dockerfile            # UI Docker imajı
│   ├── docker-compose.yaml   # UI container tanımı
│   └── requirements-ui.txt  # UI Python bağımlılıkları
│
├── remote_server_files/
│   ├── main.py               # FastAPI TTS sunucusu
│   ├── Dockerfile            # TTS sunucu Docker imajı
│   ├── docker-compose.yaml   # TTS container tanımı
│   └── requirements-server.txt
│
└── tmp/
    ├── sample_rag_input.csv           # Örnek soru-cevap CSV
    ├── askeri_egitim_kitabi.pdf       # Örnek indekslenecek PDF
    └── taktik_muharebe_yarali_bakimi_el_kitabi.pdf
```

---

## 6. Sorun Giderme

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

---