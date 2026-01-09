import customtkinter as ctk
import tkinter as tk
import threading
from datetime import datetime, timedelta
from openai import OpenAI
import mysql.connector
import re
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors

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

        # UI Çevirisi: Placeholder
        self.input_entry = ctk.CTkEntry(input_frame, placeholder_text="Type your message here...")
        self.input_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.input_entry.bind("<Return>", lambda event: self._send_message_thread())

        # UI Çevirisi: Button
        self.send_button = ctk.CTkButton(input_frame, text="Send", width=70, command=self._send_message_thread)
        self.send_button.grid(row=0, column=1, sticky="e")

        # UI Çevirisi: Status Label
        self.status_label = ctk.CTkLabel(self.parent, text="🤖 Assistant Offline", text_color="gray")
        self.status_label.grid(row=2, column=0, padx=10, pady=(0, 5), sticky="w")

    def _init_groq_thread(self):
        threading.Thread(target=self._init_groq, daemon=True).start()

    def _init_groq(self):
        if not self.api_key:
            self._update_status("ERROR: API Key Missing", "red")
            return
        try:
            self.client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=self.api_key)
            self.client.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": "test"}], max_tokens=5)
            self.has_api = True
            self._update_status(f"🤖 Groq Ready ({self.model_name})", "green")
            self._safe_append("System", f"Groq AI initialized. Classic models active.")
            threading.Thread(target=self._preload_forecast, daemon=True).start()
        except Exception as e:
            self._update_status("API Connection Error", "red")
            self.has_api = False

    def _update_status(self, text, color):
        self.parent.after(0, lambda: self.status_label.configure(text=text, text_color=color))

    def _generate_with_retry(self, prompt, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a library assistant. Give short, technical, and concise answers."
                                                      " Avoid unnecessary polite phrases. State the information and move on."
                                                      " Answer in the language the question was asked in."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200
                )
                return response.choices[0].message.content
            except Exception as e:
                if "429" in str(e): time.sleep(10); continue
                return f"Error: {str(e)}"
        return "No response received."

    def _send_message_thread(self):
        msg = self.input_entry.get().strip()
        if not msg: return
        self.input_entry.delete(0, tk.END)
        self.input_entry.configure(state="disabled")
        self.send_button.configure(state="disabled", text="...")
        self._safe_append("You", msg)
        threading.Thread(target=self._process_and_reply, args=(msg,), daemon=True).start()

    def _process_and_reply(self, user_msg):
        if not self.has_api:
            self._safe_append("System", "No connection.")
            self.parent.after(0, self._re_enable_input)
            return
        try:
            # HİBRİT TAHMİN MANTIĞI
            forecast_data = self._handle_advanced_forecast(user_msg)
            live_occ = self._get_live_occupancy_total()

            # --- KRİTİK DÜZELTME: Context Injection İngilizceye Çevrildi ---
            context = f"Current Time: {datetime.now().strftime('%H:%M')}. Capacity: {self.capacity}. Live Occupancy: {live_occ}. "
            if forecast_data: context += f"\nForecast Analysis: {forecast_data}"

            prompt = f"Context: {context}\nUser: {user_msg}\nAnswer:"
            # -------------------------------------------------------------

            self._safe_append("ASSISTANT", self._generate_with_retry(prompt))
        except Exception as e:
            self._safe_append("System", f"Error: {str(e)[:40]}")
        self.parent.after(0, self._re_enable_input)

    def _handle_advanced_forecast(self, user_msg):
        """Saat/Gün tespiti yapar. Gelişmiş modeller hata verirse basit ortalamaya (Plan C) geçer."""
        user_msg_lower = user_msg.lower()
        gunler = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6}

        # 1. SAAT TESPİTİ (GÜNCELLENDİ: Daha esnek regex)
        # "12", "12:00", "12pm", "at 12" gibi formatları yakalar.
        # Kelime sonuna (\b) bakar, böylece "tomorrow" yazsan bile "12"yi alır.
        hour_match = re.search(r'\b(\d{1,2})(?:[:.](\d{2}))?(?:\s*(am|pm))?\b', user_msg_lower)

        target_hour = None
        if hour_match:
            try:
                raw_hour = int(hour_match.group(1))
                # 24 saati geçerse (örn: 2025 yılı gibi) saat değildir, yoksay.
                if 0 <= raw_hour <= 23:
                    target_hour = raw_hour
                    # PM kontrolü (Örn: 2 pm -> 14)
                    is_pm = hour_match.group(3) == 'pm'
                    if is_pm and target_hour < 12:
                        target_hour += 12
                    elif str(hour_match.group(3)) == 'am' and target_hour == 12:
                        target_hour = 0
            except:
                pass

        # 2. GÜN TESPİTİ (Tomorrow/Today Mantığı)
        current_weekday = datetime.now().weekday()
        target_day = None

        if "tomorrow" in user_msg_lower:
            target_day = (current_weekday + 1) % 7
        elif "today" in user_msg_lower:
            target_day = current_weekday
        else:
            # Gün ismi geçiyor mu?
            target_day = next((idx for gun, idx in gunler.items() if gun in user_msg_lower), None)

        # EĞER BELİRLİ BİR SAAT VE GÜN TESPİT EDİLDİYSE
        if target_hour is not None and target_day is not None:
            days_en = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_name = days_en[target_day]

            # PLAN A: Gelişmiş Modelleri (ForecastingEngine) Dene
            try:
                best_model, pred, err, low, high, all_res = self.forecaster.run_best_slot_forecast(
                    self.data_manager.hourly_data, target_day, target_hour, exam_mode=0
                )
                return (f"Forecast for {day_name} at {target_hour}:00 is approx {pred:.0f} people. "
                        f"(Model: {best_model}, Range: {low:.0f}-{high:.0f})")
            except Exception:
                pass  # Plan A başarısızsa Plan B'ye geç

            # PLAN B: Prophet Cache (Varsa)
            if self.forecast_cache is not None:
                try:
                    fc = self.forecast_cache.copy()
                    fc['weekday'] = fc['ds'].dt.weekday
                    fc['hour'] = fc['ds'].dt.hour
                    match = fc[(fc['weekday'] == target_day) & (fc['hour'] == target_hour)]
                    if not match.empty:
                        pred_val = match['yhat'].iloc[0]
                        return f"AI models (Prophet) predict around {pred_val:.0f} people for {day_name} at {target_hour}:00."
                except:
                    pass

            # PLAN C: (SON ÇARE) Basit Tarihsel Ortalama
            # Eğer karmaşık modeller ve Prophet çalışmazsa, elimizdeki ham verinin ortalamasını al.
            try:
                df = self.data_manager.hourly_data
                # İlgili gün ve saatteki tüm geçmiş verileri filtrele
                filtered = df[(df['weekday'] == target_day) & (df['hour'] == target_hour)]
                if not filtered.empty:
                    avg_val = filtered['saatlik_ortalama_doluluk'].mean()
                    return f"Complex models insufficient, but historical average for {day_name} {target_hour}:00 is ~{avg_val:.0f} people."
            except:
                pass

            return f"Insufficient historical data to make a prediction for {day_name} at {target_hour}:00."

        # GENEL YOĞUNLUK SORULARI
        if any(w in user_msg_lower for w in ["busy", "crowded", "week", "peak", "forecast"]):
            return self._get_prophet_peak_forecast()

        return ""

    def _get_prophet_peak_forecast(self):
        if self.forecast_cache is None: return "Weekly general trend has not been analyzed yet."
        max_row = self.forecast_cache.loc[self.forecast_cache['yhat'].idxmax()]
        # Gün adları İngilizce
        gun_adlari = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        return f"Prophet Analysis: Weekly peak is around {gun_adlari[max_row['ds'].weekday()]} {max_row['ds'].strftime('%H:%M')} ({max_row['yhat']:.0f} people)."

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
            now = datetime.now().replace(minute=0, second=0, microsecond=0)
            df = self.forecaster.run_prophet_weekly(
                self.data_manager.hourly_data,
                exam_mode=0,
                target_start_date=now
            )
            if df is not None: self.forecast_cache = df
        except:
            pass

    def _re_enable_input(self):
        self.input_entry.configure(state="normal")
        self.send_button.configure(state="normal", text="Send")
        self.input_entry.focus()

    def _safe_append(self, sender, message):
        self.parent.after(0, lambda: self._append_message_gui(sender, message))

    def _append_message_gui(self, sender, message):
        self.history_box.configure(state="normal")
        tag = "user" if sender == "You" else ("sys" if sender == "System" else "ai")

        if tag == "user":
            color = "white"
        elif tag == "sys":
            color = "#FF6B6B"
        else:
            color = "#69F0AE"
        self.history_box.tag_config(tag, foreground=color)

        self.history_box.insert("end", f"\n[{datetime.now().strftime('%H:%M')}] {sender.upper()}:\n{message}\n", tag)
        self.history_box.see("end")
        self.history_box.configure(state="disabled")