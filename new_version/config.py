import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("HATA: .env dosyası bulunamadı veya içinde GOOGLE_API_KEY tanımlı değil!")