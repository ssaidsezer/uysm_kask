env güncelle

modelleri container a indirmen gerekli.

.env in yapısına göre güncelle
dockerda kaldırınca arayüz url = "http://localhost:8501"


1) Dosyaları kopyala
UI bilgisayarına proje dosyalarını kopyala (en kolayı tüm proje klasörü):

uysm-kask/ klasörü komple
Sadece ui_files taşımak şu an yeterli değil.

2) UI bilgisayarında .env ayarla
uysm-kask/.env dosyasını aç ve özellikle bunu güncelle:

VOICE_API_URL=http://<TTS_SUNUCU_IP>:8000
Örnek:

VOICE_API_URL=http://192.168.1.50:8000
Ayrıca gerekiyorsa:

OLLAMA_*
QDRANT_*
OPENAI_API_KEY
değerlerini kendi ağındaki gerçek adreslerle güncelle.

3) Docker Desktop’ı aç
UI bilgisayarında Docker Desktop açık olsun.

4) Terminal aç ve UI compose klasörüne git
cd "C:\...\uysm-kask\ui_files"
5) UI container’ı build + run et
docker compose up -d --build
Bu komut streamlit-ui container’ını ayağa kaldırır.

6) Çalıştığını kontrol et
docker compose ps
docker logs --tail 100 streamlit-ui
Arayüz:

http://localhost:8501
7) TTS bağlantısını doğrula
UI’de TTS sekmesinden kısa bir metin dene.

Hata olursa en hızlı kontrol:

.env içindeki VOICE_API_URL doğru mu?
UI bilgisayarından TTS sunucuya erişim var mı? (firewall/ağ)
TTS sunucuda http://<IP>:8000/health dönüyor mu?
İstersen bir sonraki mesajda sana kopyala-yapıştır tek checklist formatında (komut + beklenen çıktı) veririm.

TTS MODELLERİNİ UZAK BİLGİSAYARA İNDİR!!! SONRA ENV GÜNCELLE