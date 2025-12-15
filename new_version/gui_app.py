import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
import threading
from ai_assistant import LibraryChatbot
from config import GOOGLE_API_KEY

try:
    from tkcalendar import DateEntry

    HAS_TKCALENDAR = True
except ImportError:
    HAS_TKCALENDAR = False

try:
    from prophet import Prophet

    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False


class LibTrackApp(ctk.CTk):
    def __init__(self, data_manager, forecasting_engine):
        super().__init__()
        self.data_manager = data_manager
        self.forecaster = forecasting_engine

        self.title("LibTrack AI - Kütüphane Doluluk Tahmini")
        self.geometry("1100x800")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Görsel harita için ışık listeleri (Masa A ve Masa B ayrı)
        self.lights_masa_a = []
        self.lights_masa_b = []

        # Chat penceresinin açık/kapalı durumu
        self.is_chat_open = False

        self._init_ui()

        # Canlı veriyi başlat
        self.update_live_occupancy(initial_run=True)

        # Prophet'i başlat (Artık arayüzü dondurmayacak)
        self.initial_prophet_run()

    def _init_ui(self):
        # --- ANA İÇERİK ---
        self.scrollable_main_frame = ctk.CTkScrollableFrame(self)
        self.scrollable_main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.scrollable_main_frame.grid_columnconfigure(0, weight=1)

        # Başlık
        title_lbl = ctk.CTkLabel(self.scrollable_main_frame, text="Kütüphane Doluluk Tahmin Motoru",
                                 font=ctk.CTkFont(size=24, weight="bold"))
        title_lbl.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")

        # Canlı Veri ve Harita Bölümü
        self._setup_live_data_section()

        # Sekmeler
        self.nb = ttk.Notebook(self.scrollable_main_frame)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background=self._apply_appearance_mode(ctk.ThemeManager.theme["CTk"]["fg_color"]))
        style.configure("TNotebook.Tab",
                        background=self._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"]),
                        padding=[10, 5])
        style.map("TNotebook.Tab", background=[("selected", ctk.ThemeManager.theme["CTkButton"]["fg_color"][1])])
        self.nb.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")

        # Sekmeleri Ekle
        self.f_slot = ctk.CTkFrame(self.nb)
        self.nb.add(self.f_slot, text=" 🎯 Tek Slot Tahmini ")
        self.setup_slot_prediction_tab(self.f_slot)

        self.f_prophet = ctk.CTkFrame(self.nb)
        self.nb.add(self.f_prophet, text=" 📊 Haftalık Genel Tahmin ")
        self.setup_prophet_tab(self.f_prophet)

        # --- FLOATING CHAT (YÜZEN SOHBET) ---
        self._setup_floating_chat()

    def _setup_floating_chat(self):
        self.chat_window = ctk.CTkFrame(self, width=350, height=500, corner_radius=15,
                                        fg_color=("white", "gray20"), border_width=2, border_color="gray")

        self.chat_window.grid_columnconfigure(0, weight=1)
        self.chat_window.grid_rowconfigure(1, weight=1)

        # Header
        header_frame = ctk.CTkFrame(self.chat_window, height=40, corner_radius=10, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        lbl_title = ctk.CTkLabel(header_frame, text="💬 AI Asistan", font=("Arial", 14, "bold"))
        lbl_title.pack(side="left", padx=10)

        btn_close = ctk.CTkButton(header_frame, text="✕", width=30, height=30,
                                  fg_color="transparent", text_color="red", hover_color="gray90",
                                  font=("Arial", 14, "bold"),
                                  command=self.toggle_chat)
        btn_close.pack(side="right", padx=5)

        # Content
        self.chat_content_frame = ctk.CTkFrame(self.chat_window, fg_color="transparent")
        self.chat_content_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))

        # LibraryChatbot Başlat
        self.chatbot = LibraryChatbot(
            parent_frame=self.chat_content_frame,
            api_key=GOOGLE_API_KEY,
            db_config=self.data_manager.db_config,
            capacity=self.forecaster.capacity,
            data_manager=self.data_manager,  # Data manager eklendi
            forecaster=self.forecaster  # Forecaster eklendi
        )

        # Toggle Button
        self.btn_chat_toggle = ctk.CTkButton(
            self,
            text="💬",
            width=60,
            height=60,
            corner_radius=30,
            font=("Arial", 30),
            fg_color="#3B8ED0",
            hover_color="#36719F",
            command=self.toggle_chat
        )
        self.btn_chat_toggle.place(relx=0.98, rely=0.98, anchor="se")

    def toggle_chat(self):
        if self.is_chat_open:
            self.chat_window.place_forget()
            self.btn_chat_toggle.configure(text="💬")
        else:
            self.chat_window.place(relx=0.98, rely=0.90, anchor="se")
            self.btn_chat_toggle.configure(text="🔽")
            self.chat_window.lift()
        self.is_chat_open = not self.is_chat_open

    def _setup_live_data_section(self):
        self.live_data_frame = ctk.CTkFrame(self.scrollable_main_frame, fg_color="transparent")
        self.live_data_frame.grid(row=1, column=0, padx=10, pady=(0, 5), sticky="ew")

        self.live_data_frame.grid_columnconfigure(0, weight=1)
        self.live_data_frame.grid_columnconfigure(1, weight=0)
        self.live_data_frame.grid_columnconfigure(2, weight=0)

        self.card_live = self.create_result_card(self.live_data_frame, "🔴 ANLIK DOLULUK (Canlı DB)", "DB Bağlanıyor...",
                                                 0, start_row=0)

        map_container = ctk.CTkFrame(self.live_data_frame, fg_color="white", corner_radius=10)
        map_container.grid(row=0, column=1, padx=10, pady=10, sticky="ns")

        self.map_canvas = tk.Canvas(map_container, width=220, height=100, bg="white", highlightthickness=0)
        self.map_canvas.pack(padx=5, pady=5)
        self._draw_seat_map()

        self.btn_live_update = ctk.CTkButton(
            self.live_data_frame,
            text="🔄 Canlı Veriyi\nGüncelle",
            command=self.update_live_occupancy,
            height=80,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.btn_live_update.grid(row=0, column=2, padx=10, pady=5, sticky="e")

    def _draw_seat_map(self):
        """Masa A ve Masa B'yi ayrı ayrı çizer ve HER MASA İÇİN TEK BİR NOKTA (Işık) koyar."""
        # Haritayı temizle
        self.map_canvas.delete("all")
        self.lights_masa_a = []
        self.lights_masa_b = []

        # --- MASA A (Kamera 0) ---
        self.map_canvas.create_rectangle(20, 30, 90, 80, fill="#d0d0d0", outline="gray")
        self.map_canvas.create_text(55, 55, text="Masa A\n(Cam 0)", font=("Arial", 9, "bold"), fill="gray")

        # Masa A için TEK Işık (Üst Ortada)
        x_a, y_a = 55, 20
        # Biraz daha büyük (radius=8) tek bir nokta çiziyoruz
        oval_id_a = self.map_canvas.create_oval(x_a - 8, y_a - 8, x_a + 8, y_a + 8, fill="gray", outline="black")
        self.lights_masa_a.append(oval_id_a)

        # --- MASA B (IP Kamera) ---
        self.map_canvas.create_rectangle(130, 30, 200, 80, fill="#d0d0d0", outline="gray")
        self.map_canvas.create_text(165, 55, text="Masa B\n(IP Cam)", font=("Arial", 9, "bold"), fill="gray")

        # Masa B için TEK Işık (Üst Ortada)
        x_b, y_b = 165, 20
        # Biraz daha büyük (radius=8) tek bir nokta çiziyoruz
        oval_id_b = self.map_canvas.create_oval(x_b - 8, y_b - 8, x_b + 8, y_b + 8, fill="gray", outline="black")
        self.lights_masa_b.append(oval_id_b)

    def _update_map_visuals(self, occupancy_data):
        """
        occupancy_data: {'0': 1, 'http://...': 0} gibi bir sözlük gelir.
        Buna göre masa renklerini günceller.
        """
        # 1. Masa A (Kamera "0" kontrolü)
        count_a = occupancy_data.get('0', 0)  # Eğer '0' yoksa 0 varsay
        # Doluysa (>0) Kırmızı, Boşsa Yeşil
        color_a = "red" if count_a > 0 else "#00ff00"

        for light_id in self.lights_masa_a:
            self.map_canvas.itemconfig(light_id, fill=color_a)

        # 2. Masa B (Diğer tüm kameralar - IP Kamera)
        # '0' olmayan herhangi bir key varsa onu Masa B kabul ediyoruz
        count_b = 0
        for key, val in occupancy_data.items():
            if key != '0':
                count_b = val
                break  # İlk bulduğu harici kamerayı alır

        color_b = "red" if count_b > 0 else "#00ff00"

        for light_id in self.lights_masa_b:
            self.map_canvas.itemconfig(light_id, fill=color_b)

    def create_result_card(self, parent, title, value, col, start_row=0):
        card = ctk.CTkFrame(parent, fg_color="white")
        card.grid(row=start_row, column=col, padx=10, pady=10, sticky="nsew")

        title_lbl = ctk.CTkLabel(card, text=title, font=("Arial", 12))
        title_lbl.pack(padx=10, pady=(10, 0), anchor="w")

        value_lbl = ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=28, weight="bold"))
        value_lbl.pack(padx=10, pady=(0, 10), anchor="center")
        return value_lbl

    def update_live_occupancy(self, initial_run=False):
        # Canlı veriyi de thread içine alabiliriz ama genellikle hızlıdır.
        # Yine de UI'ı bloklamaması için 'threading' içine alalım.
        threading.Thread(target=self._live_occupancy_worker, args=(initial_run,), daemon=True).start()

    def _live_occupancy_worker(self, initial_run):
        # Bu fonksiyon arka planda çalışır
        if not initial_run:
            self.after(0, lambda: self.card_live.configure(text_color="gray", text="Yükleniyor..."))

        # Sözlük döner: {'0': 1, 'http://...': 0}
        occupancy_data = self.data_manager.fetch_live_occupancy()

        # UI Güncellemesini ana thread'e geri yolla
        self.after(0, lambda: self._update_live_ui(occupancy_data, initial_run))

    def _update_live_ui(self, occupancy_data, initial_run):
        if isinstance(occupancy_data, dict):
            # Toplam sayıyı hesapla
            total_count = sum(occupancy_data.values())
            capacity = self.forecaster.capacity
            perc = 100 * total_count / capacity if capacity > 0 else 0

            self.card_live.configure(text_color="black", text=f"{total_count} Kişi ({perc:.1f}%)")

            # Harita renklerini güncelle
            self._update_map_visuals(occupancy_data)

            if not initial_run:
                masa_a_val = occupancy_data.get('0', 0)
                masa_b_val = sum(v for k, v in occupancy_data.items() if k != '0')
                messagebox.showinfo("Başarılı",
                                    f"Canlı doluluk güncellendi.\nMasa A: {masa_a_val}\nMasa B: {masa_b_val}")
        else:
            # Hata mesajı string olarak geldiyse
            self.card_live.configure(text_color="red", text="Veri Yok")
            print(f"Hata detayı: {occupancy_data}")

    def setup_slot_prediction_tab(self, parent_frame):
        parent_frame.grid_columnconfigure(0, weight=1)
        parent_frame.grid_rowconfigure(3, weight=1)

        control_frame = ctk.CTkFrame(parent_frame)
        control_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        control_frame.grid_columnconfigure((0, 2), weight=1)

        input_subframe = ctk.CTkFrame(control_frame, fg_color="transparent")
        input_subframe.grid(row=0, column=0, rowspan=2, padx=(10, 20), pady=10, sticky="nsw")

        ctk.CTkLabel(input_subframe, text="1. Tahmin Tarihi:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        if HAS_TKCALENDAR:
            self.date_entry_frame = ctk.CTkFrame(input_subframe, fg_color="transparent", width=150)
            self.date_entry_frame.grid(row=0, column=1, padx=10, pady=5, sticky="w")
            self.date_entry = DateEntry(
                self.date_entry_frame, width=12, date_pattern="yyyy-mm-dd",
                mindate=self.data_manager.min_date, font=("Arial", 12)
            )
            self.date_entry.set_date(self.data_manager.max_date)
            self.date_entry.pack(fill='x', expand=True, padx=2)
        else:
            self.date_entry = ctk.CTkEntry(input_subframe, width=150)
            self.date_entry.insert(0, "2025-12-25")
            self.date_entry.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        ctk.CTkLabel(input_subframe, text="2. Saat Slotu:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        slots = [f"{h:02d}:00-{h + 1:02d}:00" for h in range(8, 23)]
        self.slot_combo = ctk.CTkComboBox(input_subframe, values=slots, width=150, state="readonly")
        self.slot_combo.set(slots[4])
        self.slot_combo.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        self.exam_var = tk.IntVar(value=0)
        self.exam_check = ctk.CTkCheckBox(
            input_subframe, text="3. Sınav Dönemi Deseni Kullan", variable=self.exam_var, onvalue=1, offvalue=0
        )
        self.exam_check.grid(row=2, column=0, columnspan=2, padx=10, pady=(5, 10), sticky="w")

        self.btn = ctk.CTkButton(control_frame, text="🚀 Tek Slot Tahminini Çalıştır", command=self.make_slot_forecast,
                                 font=ctk.CTkFont(size=15, weight="bold"), height=50)
        self.btn.grid(row=0, column=1, rowspan=2, padx=20, pady=10, sticky="nsew")

        self.result_frame = ctk.CTkFrame(parent_frame, fg_color="transparent")
        self.result_frame.grid(row=1, column=0, padx=20, pady=(10, 10), sticky="ew")
        self.result_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.result_frame.grid_rowconfigure(1, weight=1)

        self.status_lbl = ctk.CTkLabel(self.result_frame, text="Durum: Tahmin Yapılmadı",
                                       font=ctk.CTkFont(size=18, weight="bold"), anchor="w")
        self.status_lbl.grid(row=0, column=0, columnspan=3, padx=10, pady=(5, 5), sticky="w")

        self.card_pred = self.create_result_card(self.result_frame, "👥 Tahmin (Kişi)", "N/A", 0, start_row=1)
        self.card_perc = self.create_result_card(self.result_frame, "📈 Doluluk Oranı", "N/A", 1, start_row=1)
        self.card_interval = self.create_result_card(self.result_frame, "🔒 Güven Aralığı (~95%)", "N/A", 2, start_row=1)

        self.detail_frame = ctk.CTkFrame(parent_frame)
        self.detail_frame.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.detail_frame.grid_columnconfigure(0, weight=1)
        self.detail_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(self.detail_frame, text="🔍 Detaylar ve Model Performansı:",
                     font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")

        self.result_text = ctk.CTkTextbox(self.detail_frame, height=180, width=650)
        self.result_text.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="nsew")
        self.result_text.insert("0.0", "Tahmini çalıştırmak için yukarıdaki butonu kullanın...")
        self.result_text.configure(state="disabled")

    def make_slot_forecast(self):
        # Bu işlem genelde hızlıdır (pandas işlemi), thread'e gerek yok ama
        # yine de "Hesaplanıyor" yazısını görmek için
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.card_pred.configure(text="Hesaplanıyor...")
        self.status_lbl.configure(text="Durum: Hesaplanıyor...", text_color="gray")
        self.update_idletasks()  # UI'ı zorla güncelle

        try:
            if HAS_TKCALENDAR:
                date = pd.to_datetime(self.date_entry.get_date())
            else:
                date_str_in = self.date_entry.get().strip()
                if not date_str_in: raise ValueError("Lütfen geçerli bir tarih girin.")
                date = pd.to_datetime(date_str_in)

            date_str = date.strftime("%Y-%m-%d")
            weekday = date.weekday()
            slot_str = self.slot_combo.get().strip()
            start_hour = int(slot_str.split(":")[0])
            exam_mode = self.exam_var.get()

            best_model, best_pred, best_err, low, high, all_results = self.forecaster.run_best_slot_forecast(
                self.data_manager.hourly_data, weekday, start_hour, exam_mode
            )

            perc = 100 * best_pred / self.forecaster.capacity
            perc_low = 100 * low / self.forecaster.capacity
            perc_high = 100 * high / self.forecaster.capacity

            status, status_color = self._determine_status(perc)

            self.card_pred.configure(text=f"{best_pred:.1f} Kişi")
            self.card_perc.configure(text=f"{perc:.1f}%")
            self.card_interval.configure(text=f"[{low:.1f} - {high:.1f}]")
            self.status_lbl.configure(text=f"Durum: {status}", text_color=status_color)

            self._write_slot_report(date_str, weekday, slot_str, exam_mode, best_model, best_err, perc_low, perc_high,
                                    all_results)

        except ValueError as e:
            self._handle_error(str(e), "Giriş Hatası")
        except Exception as e:
            self._handle_error(str(e), "Tahmin Hatası")
        finally:
            self.result_text.configure(state="disabled")

    def _determine_status(self, perc):
        if perc > 80:
            return "YÜKSEK (Yoğunluk bekleniyor)", "red"
        elif perc > 50:
            return "ORTA (Yer bulmak zorlaşabilir)", "orange"
        else:
            return "DÜŞÜK (Rahat çalışma ortamı)", "green"

    def _write_slot_report(self, date_str, weekday, slot_str, exam_mode, best_model, best_err, perc_low, perc_high,
                           all_results):
        gun_adlari = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']
        output = f"📅 Tarih: {date_str} ({gun_adlari[weekday]})\n"
        output += f"⏰ Slot: {slot_str}\n"
        output += f"📚 Dönem: {'Sınav Dönemi Verisi' if exam_mode == 1 else 'Normal Dönem Verisi'}\n"
        output += "--------------------------------------------------------\n"
        output += f"✨ En İyi Model: {best_model}\n"
        output += f"🎯 Model Hata Payı (MAE): {best_err:.2f}\n\n"
        output += f"⚖️ KESİN ARALIK DETAYI:\n   - % Aralığı: {perc_low:.1f}% - {perc_high:.1f}%\n\n"
        output += "📊 Tüm Modellerin Hata Değerleri (MAE):\n"
        for name, (pred, err) in all_results.items():
            suffix = "  <-- Seçilen Model" if name == best_model else ""
            output += f"  - {name}: {err:.2f}{suffix}\n"
        self.result_text.insert(tk.END, output)

    def _handle_error(self, msg, title):
        messagebox.showerror(title, msg)
        self.card_pred.configure(text="HATA!")
        self.status_lbl.configure(text="Durum: HATA!", text_color="red")
        self.result_text.insert(tk.END, f"Hata: {msg}")

    # PROPHET

    def setup_prophet_tab(self, parent_frame):
        parent_frame.grid_columnconfigure(0, weight=1)
        parent_frame.grid_rowconfigure(3, weight=1)

        input_frame = ctk.CTkFrame(parent_frame)
        input_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")

        ctk.CTkLabel(input_frame, text="Prophet Haftalık Tahmin Ayarları:",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(padx=10, pady=5, anchor="w")

        self.prophet_exam_var = tk.IntVar(value=0)
        self.prophet_exam_check = ctk.CTkCheckBox(input_frame, text="Sınav Dönemi Verisi Kullan",
                                                  variable=self.prophet_exam_var)
        self.prophet_exam_check.pack(padx=10, pady=5, anchor="w")

        self.btn_prophet = ctk.CTkButton(input_frame, text="♻️ Haftalık Tahmini Güncelle/Çalıştır",
                                         command=self.run_prophet_forecast,
                                         font=ctk.CTkFont(size=14, weight="bold"), height=30)
        self.btn_prophet.pack(padx=10, pady=10, fill="x")

        self.summary_frame = ctk.CTkFrame(parent_frame, fg_color="transparent")
        self.summary_frame.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="ew")
        self.summary_frame.grid_columnconfigure((0, 1), weight=1)

        self.peak_day_lbl = ctk.CTkLabel(self.summary_frame, text="✨ Haftanın En Yoğun Günü: ?",
                                         font=ctk.CTkFont(size=14, weight="bold"), anchor="w", text_color="orange")
        self.peak_day_lbl.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.peak_hour_lbl = ctk.CTkLabel(self.summary_frame, text="🔥 Haftanın En Yoğun Saati: ?",
                                          font=ctk.CTkFont(size=14, weight="bold"), anchor="w", text_color="red")
        self.peak_hour_lbl.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        result_frame = ctk.CTkFrame(parent_frame)
        result_frame.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="nsew")
        result_frame.grid_columnconfigure(0, weight=1)
        result_frame.grid_rowconfigure(0, weight=1)

        self.prophet_textbox = ctk.CTkTextbox(result_frame, height=450, font=("Courier New", 12))
        self.prophet_textbox.pack(fill='both', expand=True, padx=10, pady=10)
        self.prophet_textbox.configure(state="disabled")

    def initial_prophet_run(self):
        # Arka planda çalıştır (sessizce)
        self.run_prophet_forecast(silent=True)

    def run_prophet_forecast(self, silent=False):
        if not HAS_PROPHET:
            if not silent: messagebox.showerror("Hata", "Prophet kütüphanesi yüklü değil.")
            return

        # UI'da "Hesaplanıyor" göster
        self.prophet_textbox.configure(state="normal")
        self.prophet_textbox.delete("1.0", "end")
        self.prophet_textbox.insert("0.0",
                                    "Lütfen bekleyin, model hesaplanıyor (Bu işlem 5-10 saniye sürebilir, arayüz donmayacaktır)...")
        self.prophet_textbox.configure(state="disabled")

        # THREADING: Ağır işi arka plana at
        current_mode = self.prophet_exam_var.get()
        threading.Thread(target=self._prophet_worker, args=(current_mode, silent), daemon=True).start()

    def _prophet_worker(self, current_mode, silent):
        """Bu fonksiyon arka planda çalışır, arayüzü dondurmaz"""
        try:
            forecast = self.forecaster.run_prophet_weekly(self.data_manager.hourly_data, current_mode)
        except Exception as e:
            forecast = str(e)  # Hatayı string olarak taşı

        # Sonucu arayüze basmak için ana thread'e geri dön
        self.after(0, lambda: self._update_prophet_ui(forecast, silent))

    def _update_prophet_ui(self, forecast, silent):
        """Bu fonksiyon ana thread'de çalışır ve sonucu ekrana basar"""
        self.prophet_textbox.configure(state="normal")

        if isinstance(forecast, str):  # Hata mesajı geldiyse
            self.prophet_textbox.delete("1.0", "end")
            self.prophet_textbox.insert("0.0", f"Hata: {forecast}")
            self.prophet_textbox.configure(state="disabled")
            return

        if forecast is None:
            self.prophet_textbox.delete("1.0", "end")
            self.prophet_textbox.insert("0.0", "Yetersiz Veri veya Hata.")
            self.prophet_textbox.configure(state="disabled")
            return

        # Raporlama Mantığı (Aynı kalıyor)
        self._generate_prophet_report(forecast, silent)

    def _generate_prophet_report(self, forecast, silent):
        forecast['hour'] = forecast['ds'].dt.hour
        filtered = forecast[(forecast['hour'] >= 7) & (forecast['hour'] <= 23)]
        gun_adlari = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']

        if not filtered.empty:
            max_row = filtered.loc[filtered['yhat'].idxmax()]
            self.peak_day_lbl.configure(
                text=f"✨ En Yoğun Gün: {gun_adlari[max_row['ds'].weekday()]} ({max_row['yhat']:.1f} Kişi)")
            self.peak_hour_lbl.configure(
                text=f"🔥 En Yoğun Saat: {max_row['ds'].strftime('%H:00')} ({max_row['yhat']:.1f} Kişi)")

        output = ""
        header = "SAAT | TAHMİN (Kişi) | % DOLULUK | GÜVEN ARALIĞI   | DURUM\n"
        separator = "-----+---------------+-----------+-----------------+---------------\n"

        for day, group in filtered.groupby(filtered['ds'].dt.date):
            day_name = gun_adlari[group['ds'].iloc[0].weekday()]
            output += f"=================================================================================\n"
            output += f"🗓️ {day.strftime('%Y-%m-%d')} - {day_name.upper()}\n"
            output += "=================================================================================\n"
            output += header + separator

            for _, row in group.iterrows():
                perc = (row['yhat'] / self.forecaster.capacity) * 100
                durum = "🔴 YÜKSEK" if perc > 80 else ("🟡 ORTA" if perc > 50 else "🟢 DÜŞÜK")

                output += f"{row['ds'].strftime('%H:%M'):<4} | {row['yhat']:^13.1f} | {perc:^9.1f}% | {row['yhat_lower']:.1f}-{row['yhat_upper']:.1f} | {durum}\n"
            output += "\n"

        self.prophet_textbox.delete("1.0", "end")
        self.prophet_textbox.insert("0.0", output)
        self.prophet_textbox.configure(state="disabled")

        if not silent:
            messagebox.showinfo("Başarılı", "Prophet Tahmini Güncellendi.")