import pandas as pd
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
import threading
import re
from datetime import datetime

# --- GRAFİK İÇİN GEREKLİ IMPORTLAR (BUNLAR EKSİKTİ) ---
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
import mplcursors
# -----------------------------------------------------

from ai_assistant import LibraryChatbot
from config import GROQ_API_KEY

# Check for Calendar Library
try:
    from tkcalendar import DateEntry

    HAS_TKCALENDAR = True
except ImportError:
    HAS_TKCALENDAR = False

# Check for Prophet Library
try:
    from prophet import Prophet

    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

# --- VISUAL SETTINGS ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class LibTrackApp(ctk.CTk):
    def __init__(self, data_manager, forecasting_engine):
        super().__init__()

        # --- Backend Connections ---
        self.data_manager = data_manager
        self.forecaster = forecasting_engine

        # --- Window Settings ---
        self.title("LibTrack AI - Smart Library System")
        self.geometry("1200x800")

        # Map light references
        self.lights_masa_a = []
        self.lights_masa_b = []

        # Chat state
        self.is_chat_open = False

        # Main Grid Structure
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._init_sidebar()
        self._init_pages()
        self._init_floating_chat()

        # Start Page
        self.select_frame("dashboard")

        # Initialize Data
        self.update_live_occupancy(initial_run=True)
        self.initial_prophet_run()

    def _init_sidebar(self):
        """Modern navigation menu on the left"""
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        # Logo / Title
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="LibTrack AI",
                                       font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Menu Buttons
        self.btn_dashboard = self._create_sidebar_button("📊 Live Monitor", lambda: self.select_frame("dashboard"))
        self.btn_dashboard.grid(row=1, column=0, padx=20, pady=10)

        self.btn_slot = self._create_sidebar_button("🎯 Slot Forecast", lambda: self.select_frame("slot"))
        self.btn_slot.grid(row=2, column=0, padx=20, pady=10)

        self.btn_weekly = self._create_sidebar_button("📅 Weekly Analysis", lambda: self.select_frame("weekly"))
        self.btn_weekly.grid(row=3, column=0, padx=20, pady=10)

        # Footer Info
        self.lbl_version = ctk.CTkLabel(self.sidebar_frame, text="v2.5 Stable", text_color="gray50")
        self.lbl_version.grid(row=5, column=0, padx=20, pady=20)

    def _create_sidebar_button(self, text, command):
        return ctk.CTkButton(self.sidebar_frame, text=text, command=command,
                             fg_color="transparent", text_color=("gray10", "gray90"),
                             hover_color=("gray70", "gray30"), anchor="w", height=40,
                             font=ctk.CTkFont(size=14, weight="bold"))

    def _init_pages(self):
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)

        self.page_dashboard = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self._setup_dashboard_ui(self.page_dashboard)

        self.page_slot = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self._setup_slot_ui(self.page_slot)

        self.page_weekly = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self._setup_weekly_ui(self.page_weekly)

    def select_frame(self, name):
        for btn in [self.btn_dashboard, self.btn_slot, self.btn_weekly]:
            btn.configure(fg_color="transparent")

        self.page_dashboard.grid_forget()
        self.page_slot.grid_forget()
        self.page_weekly.grid_forget()

        if name == "dashboard":
            self.page_dashboard.grid(row=0, column=0, sticky="nsew")
            self.btn_dashboard.configure(fg_color=("gray75", "gray25"))
        elif name == "slot":
            self.page_slot.grid(row=0, column=0, sticky="nsew")
            self.btn_slot.configure(fg_color=("gray75", "gray25"))
        elif name == "weekly":
            self.page_weekly.grid(row=0, column=0, sticky="nsew")
            self.btn_weekly.configure(fg_color=("gray75", "gray25"))

    # ==========================================
    # 1. DASHBOARD
    # ==========================================
    def _setup_dashboard_ui(self, parent):
        parent.grid_columnconfigure((0, 1), weight=1)
        parent.grid_rowconfigure(2, weight=1)
        parent.grid_rowconfigure(3, weight=0)

        ctk.CTkLabel(parent, text="Live Library Status", font=ctk.CTkFont(size=24, weight="bold")).grid(row=0, column=0,
                                                                                                        columnspan=2,
                                                                                                        sticky="w",
                                                                                                        pady=(0, 10))

        metrics_frame = ctk.CTkFrame(parent, fg_color="transparent")
        metrics_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        metrics_frame.grid_columnconfigure((0, 1), weight=1)

        self.card_total = self._create_metric_card(metrics_frame, "Current Occupancy", "...", row=0, col=0,
                                                   color="#2CC985")
        self.card_occupancy = self._create_metric_card(metrics_frame, "Occupancy Rate", "%...", row=0, col=1,
                                                       color="#3B8ED0")

        map_frame = ctk.CTkFrame(parent, corner_radius=15)
        map_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")

        ctk.CTkLabel(map_frame, text="📍 Floor Plan & Heatmap", font=ctk.CTkFont(size=16, weight="bold")).pack(
            pady=(10, 5))

        self.map_canvas = tk.Canvas(map_frame, width=600, height=220, bg="#2B2B2B", highlightthickness=0)
        self.map_canvas.pack(pady=5, fill="both", expand=True)

        self._draw_modern_map()

        self.btn_refresh = ctk.CTkButton(parent, text="🔄 Update Data Now", height=45,
                                         font=ctk.CTkFont(size=14, weight="bold"),
                                         command=self.update_live_occupancy)
        self.btn_refresh.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    def _create_metric_card(self, parent, title, value, row, col, color):
        card = ctk.CTkFrame(parent,
                            fg_color=self._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"]))
        card.grid(row=row, column=col, padx=10, pady=10, sticky="ew")

        strip = ctk.CTkFrame(card, width=10, fg_color=color, corner_radius=5)
        strip.pack(side="left", fill="y", padx=(5, 10), pady=5)

        content = ctk.CTkFrame(card, fg_color="transparent")
        content.pack(side="left", fill="both", expand=True, pady=10)

        ctk.CTkLabel(content, text=title, font=("Arial", 14), text_color="gray70").pack(anchor="w")
        value_lbl = ctk.CTkLabel(content, text=value, font=("Arial", 32, "bold"))
        value_lbl.pack(anchor="w")
        return value_lbl

    def _draw_modern_map(self):
        self.map_canvas.delete("all")
        self.lights_masa_a = []
        self.lights_masa_b = []

        for i in range(0, 800, 40):
            self.map_canvas.create_line(i, 0, i, 400, fill="#333333")
            self.map_canvas.create_line(0, i, 800, i, fill="#333333")

        offset_y = -20

        # TABLE A
        self.map_canvas.create_rectangle(55, 65 + offset_y, 255, 205 + offset_y, fill="#1a1a1a", outline="")
        self.map_canvas.create_rectangle(50, 60 + offset_y, 250, 200 + offset_y, fill="#404040", outline="#505050",
                                         width=2)
        self.map_canvas.create_text(150, 130 + offset_y, text="TABLE A\n(Cam 0)", fill="white",
                                    font=("Arial", 12, "bold"))
        glow_a = self.map_canvas.create_oval(140, 70 + offset_y, 160, 90 + offset_y, fill="#222", outline="gray")
        self.lights_masa_a.append(glow_a)

        # TABLE B
        self.map_canvas.create_rectangle(305, 65 + offset_y, 505, 205 + offset_y, fill="#1a1a1a", outline="")
        self.map_canvas.create_rectangle(300, 60 + offset_y, 500, 200 + offset_y, fill="#404040", outline="#505050",
                                         width=2)
        self.map_canvas.create_text(400, 130 + offset_y, text="TABLE B\n(IP Cam)", fill="white",
                                    font=("Arial", 12, "bold"))
        glow_b = self.map_canvas.create_oval(390, 70 + offset_y, 410, 90 + offset_y, fill="#222", outline="gray")
        self.lights_masa_b.append(glow_b)

    def _update_map_visuals(self, occupancy_data):
        color_occupied = "#FF4444"
        color_free = "#00C851"

        count_a = occupancy_data.get('0', 0)
        col_a = color_occupied if count_a > 0 else color_free
        for light in self.lights_masa_a:
            self.map_canvas.itemconfig(light, fill=col_a, outline=col_a)

        count_b = sum(v for k, v in occupancy_data.items() if k != '0')
        col_b = color_occupied if count_b > 0 else color_free
        for light in self.lights_masa_b:
            self.map_canvas.itemconfig(light, fill=col_b, outline=col_b)

    def update_live_occupancy(self, initial_run=False):
        threading.Thread(target=self._live_occupancy_worker, args=(initial_run,), daemon=True).start()

    def _live_occupancy_worker(self, initial_run):
        if not initial_run:
            self.after(0, lambda: self.card_total.configure(text="..."))
        data = self.data_manager.fetch_live_occupancy()
        self.after(0, lambda: self._update_live_ui(data, initial_run))

    def _update_live_ui(self, occupancy_data, initial_run):
        if isinstance(occupancy_data, dict):
            total = sum(occupancy_data.values())
            cap = self.forecaster.capacity
            perc = (total / cap * 100) if cap > 0 else 0

            self.card_total.configure(text=f"{total}")
            self.card_occupancy.configure(text=f"%{perc:.1f}")
            self._update_map_visuals(occupancy_data)

            if not initial_run:
                self.btn_refresh.configure(text="✅ Updated", fg_color="green")
                self.after(2000, lambda: self.btn_refresh.configure(text="🔄 Update Data Now",
                                                                    fg_color=["#3B8ED0", "#1F6AA5"]))
        else:
            self.card_total.configure(text="ERROR", text_color="red")

    # ==========================================
    # 2. SLOT FORECAST
    # ==========================================
    def _setup_slot_ui(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(parent, text="Future Slot Forecast", font=ctk.CTkFont(size=24, weight="bold")).grid(row=0,
                                                                                                         column=0,
                                                                                                         sticky="w",
                                                                                                         pady=10)

        # Input Panel
        input_frame = ctk.CTkFrame(parent)
        input_frame.grid(row=1, column=0, sticky="ew", pady=10)

        # 1. Date
        ctk.CTkLabel(input_frame, text="Select Date:", font=("Arial", 12, "bold")).grid(row=0, column=0, padx=15,
                                                                                        pady=15, sticky="w")

        if HAS_TKCALENDAR:
            self.date_entry_container = ctk.CTkFrame(input_frame, fg_color="transparent")
            self.date_entry_container.grid(row=0, column=1, padx=10, sticky="w")

            self.date_entry = DateEntry(self.date_entry_container, width=12,
                                        background='#1F6AA5', foreground='white', borderwidth=2,
                                        date_pattern='yyyy-mm-dd', headersbackground="#144f7d",
                                        normalbackground="#333333", normalforeground="white")
            self.date_entry.pack()
        else:
            # Fallback
            self.date_entry = ctk.CTkEntry(input_frame, placeholder_text="YYYY-MM-DD")
            self.date_entry.grid(row=0, column=1, padx=10, sticky="ew")

        # 2. Slot
        ctk.CTkLabel(input_frame, text="Time Slot:", font=("Arial", 12, "bold")).grid(row=0, column=2, padx=15,
                                                                                      sticky="w")
        slots = [f"{h:02d}:00-{h + 1:02d}:00" for h in range(8, 23)]
        self.slot_combo = ctk.CTkComboBox(input_frame, values=slots)
        self.slot_combo.set(slots[4])
        self.slot_combo.grid(row=0, column=3, padx=10, sticky="ew")

        # 3. Checkbox
        self.exam_var = tk.IntVar(value=0)
        self.exam_check = ctk.CTkSwitch(input_frame, text="Exam Period Mode", variable=self.exam_var)
        self.exam_check.grid(row=1, column=0, columnspan=2, padx=15, pady=15, sticky="w")

        # Button
        self.btn_forecast = ctk.CTkButton(input_frame, text="🚀 Start Analysis", command=self.make_slot_forecast,
                                          height=40)
        self.btn_forecast.grid(row=1, column=2, columnspan=2, padx=15, pady=15, sticky="ew")

        # Result Area
        result_frame = ctk.CTkFrame(parent, fg_color="#1e1e1e")
        result_frame.grid(row=2, column=0, sticky="nsew", pady=10)
        result_frame.grid_rowconfigure(0, weight=1)
        result_frame.grid_columnconfigure(0, weight=1)

        self.result_text = ctk.CTkTextbox(result_frame, font=("Consolas", 14), text_color="#00ff00", fg_color="#1e1e1e")
        self.result_text.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.result_text.insert("0.0", "> System Ready.\n> Please select date/time and click 'Start Analysis'...")
        self.result_text.configure(state="disabled")

    def make_slot_forecast(self):
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("0.0", "> Analyzing data...\n")
        self.update_idletasks()

        try:
            if HAS_TKCALENDAR:
                date = pd.to_datetime(self.date_entry.get_date())
            else:
                date_str = self.date_entry.get()
                if not date_str: raise ValueError("Please enter a date.")
                date = pd.to_datetime(date_str)

            weekday = date.weekday()
            slot_str = self.slot_combo.get()
            start_hour = int(slot_str.split(":")[0])
            exam_mode = self.exam_var.get()

            best_model, best_pred, best_err, low, high, all_results = self.forecaster.run_best_slot_forecast(
                self.data_manager.hourly_data, weekday, start_hour, exam_mode
            )

            perc = 100 * best_pred / self.forecaster.capacity

            report = f"\n=== RESULT REPORT ===\n"
            report += f"Date : {date.strftime('%Y-%m-%d')}\n"
            report += f"Time : {slot_str}\n"
            report += f"Mode : {'Exam Period' if exam_mode else 'Regular Term'}\n"
            report += f"---------------------\n"
            report += f"PREDICTED COUNT   : {best_pred:.1f}\n"
            report += f"OCCUPANCY RATE    : %{perc:.1f}\n"
            report += f"CONFIDENCE RANGE  : [{low:.1f} - {high:.1f}]\n"
            report += f"MODEL USED        : {best_model} (Error: {best_err:.2f})\n\n"
            report += "--- Other Model Results ---\n"
            for m, (p, e) in all_results.items():
                marker = "*" if m == best_model else " "
                report += f"[{marker}] {m:<30} : Pred={p:.1f}, Error={e:.2f}\n"

            self.result_text.insert("end", report)

        except Exception as e:
            self.result_text.insert("end", f"\n[ERROR] {str(e)}")
        finally:
            self.result_text.configure(state="disabled")

    # ==========================================
    # 3. WEEKLY (PROPHET - GRAFİK + RAPOR)
    # ==========================================
    def _setup_weekly_ui(self, parent):
        # Grid ayarı: Sol taraf (Metin) dar, Sağ taraf (Grafik) geniş
        parent.grid_columnconfigure(0, weight=1)  # Metin alanı
        parent.grid_columnconfigure(1, weight=2)  # Grafik alanı
        parent.grid_rowconfigure(1, weight=1)

        # --- ÜST PANEL (Header) ---
        head_frame = ctk.CTkFrame(parent, fg_color="transparent")
        head_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ctk.CTkLabel(head_frame, text="Weekly AI Analysis (Prophet)", font=ctk.CTkFont(size=20, weight="bold")).pack(
            side="left")

        self.prophet_exam_var = tk.IntVar(value=0)
        ctk.CTkSwitch(head_frame, text="Train with Exam Data", variable=self.prophet_exam_var).pack(side="right",
                                                                                                    padx=10)

        ctk.CTkButton(head_frame, text="♻️ Update Analysis", command=self.run_prophet_forecast, width=150).pack(
            side="right")

        # --- SOL PANEL: METİN RAPORU ---
        self.prophet_textbox = ctk.CTkTextbox(parent, font=("Consolas", 11))
        self.prophet_textbox.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        self.prophet_textbox.insert("0.0", "Waiting for analysis...")
        self.prophet_textbox.configure(state="disabled")

        # --- SAĞ PANEL: GRAFİK ALANI ---
        self.chart_frame = ctk.CTkFrame(parent, fg_color="#2b2b2b")  # Grafik arka planı
        self.chart_frame.grid(row=1, column=1, sticky="nsew")

        # Grafik placeholder (Boşken ne görünsün)
        self.lbl_chart_placeholder = ctk.CTkLabel(self.chart_frame, text="Chart will appear here...", text_color="gray")
        self.lbl_chart_placeholder.place(relx=0.5, rely=0.5, anchor="center")

    def initial_prophet_run(self):
        self.run_prophet_forecast(silent=True)

    def run_prophet_forecast(self, silent=False):
        if not HAS_PROPHET:
            if not silent: messagebox.showerror("Error", "Prophet library not found.")
            return

        # UI Güncelleme (Yükleniyor...)
        self.prophet_textbox.configure(state="normal")
        self.prophet_textbox.delete("1.0", "end")
        self.prophet_textbox.insert("0.0", "Computing AI Model...\nPlease wait.")
        self.prophet_textbox.configure(state="disabled")

        mode = self.prophet_exam_var.get()
        threading.Thread(target=self._prophet_worker, args=(mode, silent), daemon=True).start()

    def _prophet_worker(self, mode, silent):
        try:
            # Tahmin hesapla
            forecast = self.forecaster.run_prophet_weekly(self.data_manager.hourly_data, mode)
        except Exception as e:
            forecast = str(e)

        # Sonucu ana thread'e gönder
        self.after(0, lambda: self._update_prophet_ui(forecast, silent))

    def _update_prophet_ui(self, forecast, silent):
        self.prophet_textbox.configure(state="normal")
        self.prophet_textbox.delete("1.0", "end")

        # Hata Kontrolü
        if isinstance(forecast, str) or forecast is None:
            self.prophet_textbox.insert("0.0", f"Analysis Failed.\nDetail: {forecast}")
            self.prophet_textbox.configure(state="disabled")
            return

        # --- VERİ HAZIRLIĞI ---
        forecast['hour'] = forecast['ds'].dt.hour
        # Sadece kütüphane saatlerini al (08:00 - 22:00 arası) grafik daha temiz olsun
        filtered = forecast[(forecast['hour'] >= 8) & (forecast['hour'] <= 22)].copy()

        # 1. SOL TARAF: METİN RAPORU YAZDIRMA
        output = f"{'DAY':<10} | {'HOUR':<5} | {'PRED':<6} | {'%':<4}\n"
        output += "-" * 40 + "\n"

        days_en = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        for index, row in filtered.iterrows():
            d_name = days_en[row['ds'].weekday()]
            h_str = row['ds'].strftime('%H:%M')
            val = row['yhat']
            perc = (val / self.forecaster.capacity) * 100
            output += f"{d_name:<10} | {h_str:<5} | {val:<6.0f} | {perc:<4.0f}\n"

        self.prophet_textbox.insert("0.0", output)
        self.prophet_textbox.configure(state="disabled")

        # 2. SAĞ TARAF: GRAFİK ÇİZİMİ (Matplotlib)
        self._draw_prophet_chart(filtered)

        if not silent: messagebox.showinfo("Completed", "Weekly analysis & chart updated.")

    def _draw_prophet_chart(self, df):
        # Önce eski grafik varsa temizle
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        # Dark Mode Uyumlu Grafik Ayarları
        plt.style.use('dark_background')  # Matplotlib dark tema
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        fig.patch.set_facecolor('#2b2b2b')  # Dış çerçeve rengi (CustomTkinter ile uyumlu)
        ax.set_facecolor('#1e1e1e')  # Grafik iç rengi

        # Çizim
        line, = ax.plot(df['ds'], df['yhat'], color='#00ffcc', linewidth=2, label='Prediction')

        # Güven Aralığı (Gölge)
        ax.fill_between(df['ds'], df['yhat_lower'], df['yhat_upper'], color='#00ffcc', alpha=0.2,
                        label='Confidence Interval')

        # Eksen Ayarları
        ax.set_title("Weekly Occupancy Forecast", color="white", fontsize=12, pad=10)
        ax.set_ylabel("Person Count", color="gray")
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')

        # X Ekseni tarih formatı

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %H:%M'))
        plt.xticks(rotation=45, fontsize=8, color='silver')
        plt.yticks(fontsize=8, color='silver')

        # --- HOVER (ÜZERİNE GELİNCE GÖSTERME) ---
        cursor = mplcursors.cursor(line, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            # Mouse grafiğin neresindeyse oradaki veriyi al
            x, y = sel.target
            # Tarihi geri dönüştür (Matplotlib tarihleri float tutar)
            date_obj = mdates.num2date(x)
            # Tooltip metni
            sel.annotation.set_text(f"{date_obj.strftime('%A %H:%M')}\n👥 {y:.0f} People")
            # Tooltip stili
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)
            sel.annotation.set_color("black")

        # Grafiği Tkinter'a Gömme
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

    # ==========================================
    # 4. CHATBOT
    # ==========================================
    def _init_floating_chat(self):
        self.chat_window = ctk.CTkFrame(self, width=400, height=500, corner_radius=20, border_width=1,
                                        border_color="gray30")
        self.chat_window.grid_propagate(False)
        self.chat_window.grid_columnconfigure(0, weight=1)
        self.chat_window.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self.chat_window, height=50, corner_radius=15, fg_color="#1F6AA5")
        header.grid(row=0, column=0, sticky="ew", padx=2, pady=2)

        ctk.CTkLabel(header, text="🤖 AI Assistant", font=("Arial", 16, "bold"), text_color="white").pack(side="left",
                                                                                                         padx=15)
        ctk.CTkButton(header, text="✕", width=30, fg_color="transparent", text_color="white",
                      command=self.toggle_chat).pack(side="right", padx=5)

        chat_body = ctk.CTkFrame(self.chat_window, fg_color="transparent")
        chat_body.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.chatbot = LibraryChatbot(
            parent_frame=chat_body,
            api_key=GROQ_API_KEY,
            db_config=self.data_manager.db_config,
            capacity=self.forecaster.capacity,
            data_manager=self.data_manager,
            forecaster=self.forecaster
        )

        self.btn_chat_toggle = ctk.CTkButton(self, text="💬", width=60, height=60, corner_radius=30,
                                             font=("Arial", 24), fg_color="#1F6AA5", hover_color="#144f7d",
                                             command=self.toggle_chat)
        self.btn_chat_toggle.place(relx=0.96, rely=0.96, anchor="se")

    def toggle_chat(self):
        if self.is_chat_open:
            self.chat_window.place_forget()
            self.btn_chat_toggle.configure(text="💬")
        else:
            self.chat_window.place(relx=0.97, rely=0.88, anchor="se")
            self.btn_chat_toggle.configure(text="🔽")
            self.chat_window.lift()
        self.is_chat_open = not self.is_chat_open


if __name__ == "__main__":
    print("Please run main.py")