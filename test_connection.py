import os
from dotenv import load_dotenv
from openai import OpenAI

# .env dosyasındaki ortam değişkenlerini yüklüyoruz
load_dotenv()

# API anahtarının yüklendiğini doğruluyoruz (anahtarı ekrana basmadan)
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("HATA: OPENAI_API_KEY ortam değişkeni bulunamadı. .env dosyasını kontrol edin.")

# OpenAI istemcisini ilklendiriyoruz
client = OpenAI(api_key=api_key)

try:
    # Minimal bir test çağrısı yapıyoruz
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Ping"}
        ],
        max_tokens=10
    )
    print("✅ Bağlantı başarılı! Yanıt:", response.choices[0].message.content.strip())

except Exception as e:
    print("❌ API Bağlantı Hatası:", str(e))