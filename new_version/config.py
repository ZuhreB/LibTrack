import os
from dotenv import load_dotenv

load_dotenv()

# Gemini yerine GROQ_API_KEY kullanıyoruz
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("HATA: .env dosyası bulunamadı veya içinde GROQ_API_KEY tanımlı değil!")