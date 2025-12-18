import customtkinter as ctk
import tkinter as tk
import threading
from datetime import datetime, timedelta
from openai import OpenAI
import mysql.connector
import re
import time

class LibraryChatbot:
    def __init__(self, parent_frame, api_key, db_config, capacity, data_manager, forecaster):
        self.parent = parent_frame
        self.api_key = api_key
        self.db_config = db_config
        self.capacity = capacity
        self.data_manager = data_manager
        self.forecaster = forecaster
        self.client = None
        self.has_api = False
        self.model_name = "llama-3.3-70b-versatile"

        # Cache Değişkenleri
        self.forecast_cache = None
        self.cache_timestamp = None

        self._setup_ui()
        self.parent.after(300, self._init_groq_thread)

    def _setup_ui(self):
        self.parent.grid_columnconfigure(0, weight=1)
        self.parent.grid_rowconfigure(0, weight=1)
        self.parent.grid_rowconfigure(1, weight=0)
        self.parent.grid_rowconfigure(2, weight=0)

        self.history_box = ctk.CTkTextbox(self.parent, state="disabled", font=("Arial", 12), wrap="word")
        self.history_box.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        input_frame = ctk.CTkFrame(self.parent, fg_color="transparent")
        input_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)

        self.input_entry = ctk.CTkEntry(input_frame, placeholder_text="Buraya mesajınızı yazın...")
        self.input_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.input_entry.bind("<Return>", lambda event: self._send_message_thread())

        self.send_button = ctk.CTkButton(input_frame, text="Gönder", width=70, command=self._send_message_thread)
        self.send_button.grid(row=0, column=1, sticky="e")

        self.status_label = ctk.CTkLabel(self.parent, text="🤖 Asistan Çevrimdışı", text_color="gray")
        self.status_label.grid(row=2, column=0, padx=10, pady=(0, 5), sticky="w")

    def _init_groq_thread(self):
        threading.Thread(target=self._init_groq, daemon=True).start()

    def _init_groq(self):
        if not self.api_key:
            self._update_status("HATA: API Anahtarı Eksik", "red")
            return
        try:
            self.client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=self.api_key)
            self.client.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": "test"}], max_tokens=5)
            self.has_api = True
            self._update_status(f"🤖 Groq Hazır ({self.model_name})", "green")
            self._safe_append("Sistem", f"Groq AI başlatıldı. Klasik modeller aktif.")
            threading.Thread(target=self._preload_forecast, daemon=True).start()
        except Exception as e:
            self._update_status("API Bağlantı Hatası", "red")
            self.has_api = False

    def _update_status(self, text, color):
        self.parent.after(0, lambda: self.status_label.configure(text=text, text_color=color))

    def _generate_with_retry(self, prompt, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Sen kütüphane asistanısın. Kısa, teknik ve öz cevaplar ver. Gereksiz nezaket cümlelerinden kaçın. Veriyi söyle ve geç."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200
                )
                return response.choices[0].message.content
            except Exception as e:
                if "429" in str(e): time.sleep(10); continue
                return f"Hata: {str(e)}"
        return "Yanıt alınamadı."

    def _send_message_thread(self):
        msg = self.input_entry.get().strip()
        if not msg: return
        self.input_entry.delete(0, tk.END)
        self.input_entry.configure(state="disabled")
        self.send_button.configure(state="disabled", text="...")
        self._safe_append("Sen", msg)
        threading.Thread(target=self._process_and_reply, args=(msg,), daemon=True).start()

    def _process_and_reply(self, user_msg):
        if not self.has_api:
            self._safe_append("Sistem", "Bağlantı yok.")
            self.parent.after(0, self._re_enable_input)
            return
        try:
            # HİBRİT TAHMİN MANTIĞI
            forecast_data = self._handle_advanced_forecast(user_msg)
            live_occ = self._get_live_occupancy_total()

            context = f"Şu an: {datetime.now().strftime('%H:%M')}. Kapasite: {self.capacity}. Canlı Doluluk: {live_occ}. "
            if forecast_data: context += f"\nTahmin Analizi: {forecast_data}"

            prompt = f"Bağlam: {context}\nKullanıcı: {user_msg}\nYanıtla:"
            self._safe_append("ASİSTAN", self._generate_with_retry(prompt))
        except Exception as e:
            self._safe_append("Sistem", f"Hata: {str(e)[:40]}")
        self.parent.after(0, self._re_enable_input)

    def _handle_advanced_forecast(self, user_msg):
        """Spesifik saatlerde klasik modelleri (MA, ES, HW, SD) kullanır."""
        user_msg_lower = user_msg.lower()
        gunler = {'pazartesi': 0, 'salı': 1, 'çarşamba': 2, 'perşembe': 3, 'cuma': 4, 'cumartesi': 5, 'pazar': 6}

        # Regex ile saat yakala
        hour_match = re.search(r'(\d{1,2})[:.](\d{2})|(\d{1,2})(?=\s*(?:için|gibi|saat|de|te|$))', user_msg_lower)
        target_hour = int(hour_match.group(1) or hour_match.group(3)) if hour_match else None

        target_day = next((idx for gun, idx in gunler.items() if gun in user_msg_lower), None)

        # EĞER BELİRLİ BİR SAAT/GÜN SORULUYORSA KLASİK MODELLERİ YARIŞTIR (ÇOK HIZLI)
        if target_hour is not None and target_day is not None:
            try:
                best_model, pred, err, low, high, all_res = self.forecaster.run_best_slot_forecast(
                    self.data_manager.hourly_data, target_day, target_hour, exam_mode=0
                )
                return (f"{best_model} modeline göre {target_hour}:00 tahmini {pred:.0f} kişi. "
                        f"Alt-Üst sınır: {low:.0f}-{high:.0f}. MAE Hata Payı: {err:.2f}")
            except Exception as e: return f"Klasik motor hatası: {e}"

        # GENEL YOĞUNLUK SORULARINDA PROPHET CACHE KONTROLÜ
        if any(w in user_msg_lower for w in ["yoğun", "kalabalık", "hafta", "zirve"]):
            return self._get_prophet_peak_forecast()
        return ""

    def _get_prophet_peak_forecast(self):
        if self.forecast_cache is None: return "Haftalık genel trend henüz analiz edilmedi."
        max_row = self.forecast_cache.loc[self.forecast_cache['yhat'].idxmax()]
        gun_adlari = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']
        return f"Prophet Analizi: Haftalık zirve {gun_adlari[max_row['ds'].weekday()]} {max_row['ds'].strftime('%H:%M')} civarı ({max_row['yhat']:.0f} kişi)."

    def _get_live_occupancy_total(self):
        if not self.db_config: return 0
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            query = "SELECT person_count FROM person_logs WHERE id IN (SELECT MAX(id) FROM person_logs GROUP BY camera_id)"
            cursor.execute(query)
            res = cursor.fetchall()
            cursor.close(); conn.close()
            return sum(int(r[0]) for r in res) if res else 0
        except: return 0

    def _preload_forecast(self):
        try:
            df = self.forecaster.run_prophet_weekly(self.data_manager.hourly_data, exam_mode=0)
            if df is not None: self.forecast_cache = df
        except: pass

    def _re_enable_input(self):
        self.input_entry.configure(state="normal")
        self.send_button.configure(state="normal", text="Gönder")
        self.input_entry.focus()

    def _safe_append(self, sender, message):
        self.parent.after(0, lambda: self._append_message_gui(sender, message))

    def _append_message_gui(self, sender, message):
        self.history_box.configure(state="normal")
        tag = "user" if sender == "Sen" else ("sys" if sender == "Sistem" else "ai")
        color = "blue" if tag == "user" else ("red" if tag == "sys" else "green")
        self.history_box.tag_config(tag, foreground=color)
        self.history_box.insert("end", f"\n[{datetime.now().strftime('%H:%M')}] {sender.upper()}:\n{message}\n", tag)
        self.history_box.see("end"); self.history_box.configure(state="disabled")