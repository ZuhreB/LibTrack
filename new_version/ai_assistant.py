import customtkinter as ctk
import tkinter as tk
import threading
from datetime import datetime, timedelta
import google.generativeai as genai
from google.api_core import exceptions  # Hata türlerini yakalamak için
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
        self.model = None
        self.has_api = False
        self.model_name = ""

        # Cache Değişkenleri
        self.forecast_cache = None
        self.cache_timestamp = None

        self._setup_ui()
        # Arayüz donmasın diye Gemini'yi arka planda başlat
        self.parent.after(200, self._init_gemini_thread)

    def _setup_ui(self):
        self.parent.grid_columnconfigure(0, weight=1)
        self.parent.grid_rowconfigure(0, weight=1)
        self.parent.grid_rowconfigure(1, weight=0)
        self.parent.grid_rowconfigure(2, weight=0)

        # Mesaj Geçmişi
        self.history_box = ctk.CTkTextbox(self.parent, state="disabled", font=("Arial", 12), wrap="word")
        self.history_box.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Giriş Alanı
        input_frame = ctk.CTkFrame(self.parent, fg_color="transparent")
        input_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)

        self.input_entry = ctk.CTkEntry(input_frame, placeholder_text="Buraya mesajınızı yazın...")
        self.input_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.input_entry.bind("<Return>", lambda event: self._send_message_thread())

        self.send_button = ctk.CTkButton(input_frame, text="Gönder", width=70, command=self._send_message_thread)
        self.send_button.grid(row=0, column=1, sticky="e")

        # Durum Çubuğu
        self.status_label = ctk.CTkLabel(self.parent, text="🤖 Asistan Çevrimdışı", text_color="gray")
        self.status_label.grid(row=2, column=0, padx=10, pady=(0, 5), sticky="w")

    def _init_gemini_thread(self):
        threading.Thread(target=self._init_gemini, daemon=True).start()

    def _init_gemini(self):
        if not self.api_key:
            self._update_status("HATA: API Anahtarı Eksik", "red")
            self.has_api = False
            return

        # GÜNCELLENMİŞ MODEL LİSTESİ (En güncel ve ücretsiz modeller)
        MODEL_PREFERENCE = [
            'gemini-1.5-flash',  # En yeni, hızlı ve ücretsiz paket destekli
            'gemini-1.5-pro',  # Daha zeki model
            'gemini-1.0-pro',  # Eski stabil model (eski adıyla gemini-pro)
            'gemini-pro'  # Yedek olarak kalsın
        ]
        self.has_api = False

        try:
            genai.configure(api_key=self.api_key)

            for model_name in MODEL_PREFERENCE:
                try:
                    # Modeli oluştur
                    self.model = genai.GenerativeModel(model_name)
                    # Bağlantıyı test etmek için boş bir chat oturumu başlatmayı dene veya model ismini logla
                    self.model_name = model_name
                    self.has_api = True
                    print(f"Başarılı Model Bağlantısı: {model_name}")
                    break
                except Exception as e:
                    print(f"Model {model_name} denenirken hata: {e}")
                    continue

        except Exception as e:
            print(f"Genel API Hatası: {e}")
            self._update_status("API Bağlantı Hatası", "red")
            self.has_api = False
            return

        if self.has_api:
            self._update_status(f"🤖 Asistan Hazır ({self.model_name})", "green")
            self._safe_append("Sistem", f"AI Asistanı başlatıldı. Aktif Model: {self.model_name}")

            # Tahmin motorunu önceden ısıt
            threading.Thread(target=self._preload_forecast, daemon=True).start()
        else:
            self._update_status("Hiçbir Modele Bağlanılamadı", "red")

    def _update_status(self, text, color):
        self.parent.after(0, lambda: self.status_label.configure(text=text, text_color=color))

    def _preload_forecast(self):
        """Tahmin motorunu arka planda çalıştırıp cache'ler."""
        try:
            df = self.forecaster.run_prophet_weekly(self.data_manager.hourly_data, exam_mode=0)
            if df is not None and not df.empty:
                self.forecast_cache = df
                self.cache_timestamp = datetime.now()
            else:
                pass
        except Exception as e:
            print(f"Önyükleme hatası: {e}")

    def _send_message_thread(self):
        user_msg = self.input_entry.get().strip()
        if not user_msg:
            return

        # UI'ı kilitle
        self.input_entry.delete(0, tk.END)
        self.input_entry.configure(state="disabled")
        self.send_button.configure(state="disabled", text="...")
        self._safe_append("Sen", user_msg)

        threading.Thread(target=self._process_and_reply, args=(user_msg,), daemon=True).start()

    def _process_and_reply(self, user_msg):
        if not self.has_api:
            self._safe_append("Sistem", "AI asistanı API bağlantısı olmadığı için cevap veremiyor.")
            self.parent.after(0, self._re_enable_input)
            return

        try:
            # 1. Verileri Topla
            forecast_result = self._handle_forecast_request(user_msg)
            live_occupancy = self._get_live_occupancy_total()

            # 2. Context Oluştur
            context = f"Şu anki zaman: {datetime.now().strftime('%H:%M')}. "
            context += f"Kütüphane Kapasitesi: {self.capacity} kişi. "
            context += f"ANLIK DOLULUK (Tüm Masalar): {live_occupancy} kişi. "

            if forecast_result and forecast_result != "HATA":
                context += f"\n\nTAHMİN SİSTEMİNDEN GELEN VERİ: {forecast_result}"

            prompt = (
                f"Sen kütüphane asistanısın. Türkçe konuş. {context} "
                f"Soru: {user_msg} (Kısa, nazik ve net cevapla)"
            )

            # 3. API'ye Gönder (Retry Mekanizması ile)
            response_text = self._generate_with_retry(prompt)
            self._safe_append("ASİSTAN", response_text)

        except Exception as e:
            self._safe_append("Sistem", f"Beklenmeyen bir hata oluştu: {str(e)[:50]}...")

        self.parent.after(0, self._re_enable_input)

    def _generate_with_retry(self, prompt, max_retries=3):
        """429 Hatası veya Sunucu Hatası alırsa bekleyip tekrar dener."""
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except exceptions.ResourceExhausted:  # 429 Kota Hatası
                wait_time = 30  # Saniye
                self._safe_append("Sistem",
                                  f"⚠️ Hız sınırı (Kota). {wait_time} saniye bekleniyor... ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            except Exception as e:
                # 404 hatası gibi durumlarda bir sonraki modele geçilemez çünkü model başta seçildi.
                # Sadece loglayıp dönüyoruz.
                return f"API Cevap Hatası: {str(e)}"

        return "Üzgünüm, şu an çok yoğunum. Lütfen biraz sonra tekrar deneyin."

    def _re_enable_input(self):
        self.input_entry.configure(state="normal")
        self.send_button.configure(state="normal", text="Gönder")
        self.input_entry.focus()

    def _handle_forecast_request(self, user_msg):
        """Kullanıcının sorusunda tarih/saat varsa Prophet tahminini çeker."""
        gun_adlari = {'pazartesi': 0, 'salı': 1, 'çarşamba': 2, 'perşembe': 3, 'cuma': 4, 'cumartesi': 5, 'pazar': 6}
        user_msg_lower = user_msg.lower()

        # Saat bulma (Regex)
        hour_match = re.search(r'(\d{1,2})[:.](\d{2})|(\d{1,2})(?=\s*(?:için|gibi|saat|de|te|$))', user_msg_lower)
        target_hour = None
        if hour_match:
            # 14:00 veya 14 gibi formatları yakala
            h_str = hour_match.group(1) or hour_match.group(3)
            if h_str:
                target_hour = int(h_str)

        if target_hour is not None and (target_hour < 0 or target_hour > 23):
            target_hour = None

        # Gün bulma
        for day_name, day_index in gun_adlari.items():
            if day_name in user_msg_lower:
                return self._get_prophet_day_forecast(day_index, day_name, target_hour)

        # Genel yoğunluk sorusu
        if any(word in user_msg_lower for word in ["en yoğun", "pik", "en kalabalık", "ne zaman dolu"]):
            return self._get_prophet_peak_forecast()

        return ""

    def _get_forecast_data(self):
        """Cache kontrolü yaparak tahmin verisini getirir"""
        is_cache_valid = False
        if self.forecast_cache is not None and self.cache_timestamp is not None:
            # Cache süresi 60 dakika
            if datetime.now() - self.cache_timestamp < timedelta(minutes=60):
                is_cache_valid = True

        if is_cache_valid:
            return self.forecast_cache

        try:
            df = self.forecaster.run_prophet_weekly(self.data_manager.hourly_data, exam_mode=0)
            if df is not None:
                self.forecast_cache = df
                self.cache_timestamp = datetime.now()
            return df
        except Exception:
            return None

    def _get_prophet_day_forecast(self, day_index, day_name, target_hour=None):
        try:
            forecast_df = self._get_forecast_data()
            if forecast_df is None or forecast_df.empty: return "Tahmin verisi şu an kullanılamıyor."

            forecast_df['weekday'] = forecast_df['ds'].dt.weekday
            day_forecast = forecast_df[forecast_df['weekday'] == day_index].copy()

            if day_forecast.empty: return f"{day_name} verisi yok."

            if target_hour is not None:
                day_forecast['hour'] = day_forecast['ds'].dt.hour
                hour_forecast = day_forecast[day_forecast['hour'] == target_hour].copy()
                if hour_forecast.empty: return f"{day_name} {target_hour}:00 verisi yok."

                target_row = hour_forecast.iloc[0]
                return f"{day_name} saat {target_hour}:00 için tahmin: {target_row['yhat']:.0f} kişi."
            else:
                peak_day_row = day_forecast.loc[day_forecast['yhat'].idxmax()]
                return f"{day_name} günü en yoğun saat: {peak_day_row['ds'].strftime('%H:%M')} ({peak_day_row['yhat']:.0f} kişi)."

        except Exception as e:
            return f"Tahmin hatası: {e}"

    def _get_prophet_peak_forecast(self):
        try:
            forecast_df = self._get_forecast_data()
            if forecast_df is None or forecast_df.empty: return "Veri hazırlanıyor."

            gun_adlari_list = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']
            max_row = forecast_df.loc[forecast_df['yhat'].idxmax()]

            peak_day = gun_adlari_list[max_row['ds'].weekday()]
            peak_time = max_row['ds'].strftime('%H:%M')

            return f"Haftalık Zirve: {peak_day} {peak_time}, tahmini {max_row['yhat']:.0f} kişi."
        except Exception:
            return "Hata."

    def _get_live_occupancy_total(self):
        """
        DÜZELTME: Sadece son satırı değil, her bir kameranın son verisinin TOPLAMINI döndürür.
        Böylece Masa A + Masa B toplam sayısını chatbot doğru bilir.
        """
        if not self.db_config: return "Bilinmiyor"
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()

            # Her bir camera_id için en son atılan kaydı bulur
            query = """
                SELECT person_count 
                FROM person_logs 
                WHERE id IN (
                    SELECT MAX(id) 
                    FROM person_logs 
                    GROUP BY camera_id
                )
            """
            cursor.execute(query)
            results = cursor.fetchall()  # Örn: [(1,), (0,)]

            cursor.close()
            connection.close()

            total_count = 0
            if results:
                for row in results:
                    total_count += int(row[0])
                return total_count
            else:
                return 0

        except Exception as e:
            print(f"Chatbot DB Error: {e}")
            return "Veri Yok"

    def _safe_append(self, sender, message):
        self.parent.after(0, lambda: self._append_message_gui(sender, message))

    def _append_message_gui(self, sender, message):
        self.history_box.configure(state="normal")
        timestamp = datetime.now().strftime("%H:%M")

        if sender == "Sen":
            tag = "user"
            fmt = f"\n[{timestamp}] 👤 SEN:\n{message}\n"
            self.history_box.tag_config("user", foreground="blue")
        elif sender == "Sistem":
            tag = "sys"
            fmt = f"\n[{timestamp}] 🔧 SİSTEM:\n{message}\n"
            self.history_box.tag_config("sys", foreground="red")
        else:
            tag = "ai"
            fmt = f"\n[{timestamp}] 🤖 ASİSTAN:\n{message}\n"
            self.history_box.tag_config("ai", foreground="green")

        self.history_box.insert("end", fmt, tag)
        self.history_box.see("end")
        self.history_box.configure(state="disabled")