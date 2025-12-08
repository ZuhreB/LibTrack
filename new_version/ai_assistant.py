import customtkinter as ctk
import threading
from datetime import datetime
import google.generativeai as genai
import mysql.connector
class LibraryChatbot:
    def __init__(self, parent_frame, api_key, db_config, capacity):
        self.parent = parent_frame
        self.api_key = api_key
        self.db_config = db_config
        self.capacity = capacity
        self.model = None
        self.has_api = False
        self.model_name = ""

        # 1 Arayüzü kuruyo
        self._setup_ui()

        # 2 Bağlantıyı başlat arayüz yüklensin diye azıcık bekliyo
        self.parent.after(200, self._init_gemini_thread)

    def _setup_ui(self):
        # Grid ayarları
        self.parent.grid_columnconfigure(0, weight=1)
        self.parent.grid_rowconfigure(0, weight=1)
        self.parent.grid_rowconfigure(1, weight=0)

        # Sohbet Geçmişi
        self.history_box = ctk.CTkTextbox(self.parent, state="disabled", font=("Arial", 12))
        self.history_box.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Giriş Alanı
        input_frame = ctk.CTkFrame(self.parent, fg_color="transparent")
        input_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(input_frame, placeholder_text="Sorunuzu buraya yazın...", height=40)
        self.entry.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        self.entry.bind("<Return>", lambda event: self._on_enter_pressed())

        self.btn_send = ctk.CTkButton(input_frame, text="Gönder ➤", width=100, height=40,
                                      command=self._on_enter_pressed)
        self.btn_send.grid(row=0, column=1, sticky="e")

    def _init_gemini_thread(self):
        threading.Thread(target=self._connect_api, daemon=True).start()

    def _connect_api(self):
        self._safe_append("Sistem", "Bağlantı ve uygun model aranıyor...")

        if not self.api_key:
            self._safe_append("Sistem", "HATA: API Anahtarı eksik.")
            return

        try:
            genai.configure(api_key=self.api_key)
            # bu apı key ile hangi modelleri kullanabiliyorsam onları listeliycek
            available_models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)

            # Öncelik sırasına göre model seç (Flash en hızlısı, Pro ikinci, hiçbiri yoksa ilki)
            target_model = ""
            if 'models/gemini-1.5-flash' in available_models:
                target_model = 'models/gemini-1.5-flash'
            elif 'gemini-1.5-flash' in available_models:
                target_model = 'gemini-1.5-flash'
            elif 'models/gemini-pro' in available_models:
                target_model = 'models/gemini-pro'
            elif available_models:
                target_model = available_models[0]  # Listede ne varsa onu al
            else:
                raise Exception("Hesabınızda erişilebilir uygun bir model bulunamadı.")

            self.model_name = target_model
            self.model = genai.GenerativeModel(target_model)
            self.has_api = True

            clean_name = target_model.replace("models/", "")
            self._safe_append("Sistem", f" Bağlantı Başarılı! (Model: {clean_name})")

        except Exception as e:
            self.has_api = False
            self._safe_append("Sistem", f" Bağlantı Hatası: {str(e)}")

    def _on_enter_pressed(self):
        msg = self.entry.get().strip()
        if not msg: return

        self.entry.delete(0, "end")
        self._safe_append("Sen", msg)

        if not self.has_api:
            self._safe_append("Sistem", "API bağlı değil.")
            return

        threading.Thread(target=self._process_ai, args=(msg,), daemon=True).start()

    def _process_ai(self, user_msg):
        try:
            count = self._get_live_occupancy()
            if count == -1:
                context = "Veritabanına erişilemiyor."
            else:
                perc = (count / self.capacity) * 100
                context = f"Şu an kütüphanede {count} kişi var. Doluluk oranı %{perc:.1f}."

            prompt = (
                f"Sen bir kütüphane asistanısın. {context} "
                f"Kullanıcıya kısa ve net cevap ver. Soru: {user_msg}"
            )

            response = self.model.generate_content(prompt)
            self._safe_append("Gemini", response.text)

        except Exception as e:
            self._safe_append("Sistem", f"Hata: {e}")

    def _safe_append(self, sender, message):
        self.parent.after(0, lambda: self._append_message_gui(sender, message))

    def _append_message_gui(self, sender, message):
        self.history_box.configure(state="normal")
        timestamp = datetime.now().strftime("%H:%M")

        if sender == "Sen":
            fmt = f"\n[{timestamp}] 👤 SEN:\n{message}\n"
        elif sender == "Sistem":
            fmt = f"\n[{timestamp}] 🔧 SİSTEM:\n{message}\n"
        else:
            fmt = f"\n[{timestamp}] 🤖 ASİSTAN:\n{message}\n"

        self.history_box.insert("end", fmt)
        self.history_box.see("end")
        self.history_box.configure(state="disabled")

    def _get_live_occupancy(self):
        if not self.db_config: return -1
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT person_count FROM person_logs ORDER BY record_date DESC LIMIT 1")
            result = cursor.fetchone()
            conn.close()
            return int(result[0]) if result else 0
        except:
            return -1