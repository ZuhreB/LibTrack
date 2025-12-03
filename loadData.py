import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
from datetime import datetime, timedelta

# Prophet ve tkcalendar kütüphanelerinin yüklü olup olmadığını kontrol eder
try:
    from prophet import Prophet

    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

try:
    from tkcalendar import DateEntry

    HAS_TKCALENDAR = True
except ImportError:
    HAS_TKCALENDAR = False

# ======================
# 1. DATABASE VE CANLI VERİ AYARLARI (YENİ EKLENTİ)
# ======================

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "zuhre060",
    "database": "yolo_db"
}

try:
    # Lütfen 'pip install mysql-connector-python' ile kütüphaneyi kurun.
    import mysql.connector

    HAS_MYSQL_CONNECTOR = True
except ImportError:
    HAS_MYSQL_CONNECTOR = False


def fetch_latest_occupancy(db_config):
    """MySQL veritabanından en güncel anlık doluluk verisini çeker."""
    if not HAS_MYSQL_CONNECTOR:
        return "Bağlantı Hatası (Kütüphane Yok)"

    try:
        # DB bağlantısını kur
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        query = """
            SELECT person_count 
            FROM person_logs 
            ORDER BY record_date DESC 
            LIMIT 1
        """
        cursor.execute(query)
        result = cursor.fetchone()

        cursor.close()
        connection.close()

        if result and result[0] is not None:
            # Veriyi float'a çevirip döndür
            return float(result[0])
        else:
            return "Veri Yok"

    except mysql.connector.Error as err:
        # DB bağlantı veya sorgu hataları
        print(f"Veritabanı Hatası: {err}")
        return f"DB Hata Kodu: {err.errno} - Bağlantı Sorunu"
    except Exception as e:
        return f"Bilinmeyen Hata: {str(e)[:40]}..."


# ======================
# 2. VERİ YÜKLE / HAZIRLA (Aynı Kaldı)
# ======================

CSV_PATH = "libtrack_dataset_bounded_realistic_v2.csv"
CAPACITY = 432

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    messagebox.showerror("Hata", f"{CSV_PATH} dosyası bulunamadı. Lütfen dosyanın doğru yolda olduğundan emin olun.")
    exit()

df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
hourly = df[df["saatlik_ortalama_doluluk"].notnull()].copy()
hourly["saatlik_ortalama_doluluk"] = hourly["saatlik_ortalama_doluluk"].astype(float)
hourly["date"] = pd.to_datetime(hourly["date"])
hourly["hour"] = hourly["datetime"].dt.hour
hourly["weekday"] = hourly["datetime"].dt.weekday

min_date = hourly["date"].min().date()
max_date = hourly["date"].max().date()


# ======================
# 3. FORECAST MODELLERİ (Aynı Kaldı)
# ======================
# ... (mae, model_moving_average, model_exponential_smoothing,
# model_holt_winters_additive, model_seasonal_decomposition,
# model_prophet_weekly_forecast, forecast_for_slot fonksiyonları aynı kalmıştır.)

def mae(actual, predicted):
    actual = pd.Series(actual)
    predicted = pd.Series(predicted, index=actual.index)
    return (actual - predicted).abs().mean()


def model_moving_average(y, window=10):
    y = y.copy()
    if len(y) <= window:
        pred = y.mean()
        preds = pd.Series(pred, index=y.index)
        err = mae(y, preds)
    else:
        preds = y.rolling(window=window).mean()
        valid = preds.dropna()
        err = mae(y.loc[valid.index], valid)
        pred = valid.iloc[-1]
    return pred, err


def model_exponential_smoothing(y, alpha=0.3):
    y = y.copy().reset_index(drop=True)
    s = [y.iloc[0]]
    preds = [y.iloc[0]]
    for i in range(1, len(y)):
        s.append(alpha * y.iloc[i] + (1 - alpha) * s[i - 1])
        preds.append(s[i - 1])
    s = pd.Series(s, index=y.index)
    preds = pd.Series(preds, index=y.index)
    err = mae(y, preds)
    pred_next = s.iloc[-1]
    return pred_next, err


def model_holt_winters_additive(y, alpha=0.3, beta=0.1, gamma=0.1, m=4):
    y = y.copy().reset_index(drop=True)
    n = len(y)
    if n < 2 * m: return model_exponential_smoothing(y, alpha=alpha)
    L0 = y.iloc[:m].mean()
    T0 = (y.iloc[m:2 * m].mean() - y.iloc[:m].mean()) / m if n >= 2 * m else (y.iloc[1] - y.iloc[0]) if n > 1 else 0.0
    S = [y.iloc[i] - L0 for i in range(m)]
    L, T, fitted = [L0], [T0], [L0 + T0 + S[0]]
    for t in range(1, n):
        idx_season = (t - m) % m
        Stm = S[idx_season] if t - m >= 0 else S[t % m]
        Lt = alpha * (y.iloc[t] - Stm) + (1 - alpha) * (L[t - 1] + T[t - 1])
        Tt = beta * (Lt - L[t - 1]) + (1 - beta) * T[t - 1]
        St = gamma * (y.iloc[t] - Lt) + (1 - gamma) * Stm
        L.append(Lt);
        T.append(Tt);
        S[t % m] = St
        fitted.append(L[t - 1] + T[t - 1] + Stm)
    fitted = pd.Series(fitted, index=y.index)
    valid_idx = y.index[m:]
    err = mae(y.loc[valid_idx], fitted.loc[valid_idx])
    pred_next = L[-1] + T[-1] + S[(n - m) % m]
    return pred_next, err


def model_seasonal_decomposition(y, m=4):
    y = y.copy().reset_index(drop=True)
    n = len(y)
    if n < 2 * m: return model_exponential_smoothing(y, alpha=0.3)
    trend = y.rolling(window=m, center=True).mean()
    detrended = y - trend
    seasonal = np.zeros(m);
    counts = np.zeros(m)
    for i in range(n):
        idx = i % m
        if not np.isnan(detrended.iloc[i]):
            seasonal[idx] += detrended.iloc[i];
            counts[idx] += 1
    for i in range(m):
        if counts[i] > 0:
            seasonal[i] /= counts[i]
        else:
            seasonal[i] = 0.0
    seasonal_series = pd.Series([seasonal[i % m] for i in range(n)], index=y.index)
    recon = trend + seasonal_series
    valid_mask = ~trend.isna()
    if valid_mask.sum() < 3: return model_exponential_smoothing(y, alpha=0.3)
    err = mae(y[valid_mask], recon[valid_mask])
    last_trend = trend[valid_mask].iloc[-1]
    pred_next = last_trend + seasonal[n % m]
    return pred_next, err


def model_prophet_weekly_forecast(hourly_df, exam_mode, capacity):
    """Tüm geçmiş veriyi (saatlik) kullanarak, ileri 7 günlük (168 saat) tahmin yapar."""
    if not HAS_PROPHET:
        return None

    df_full = hourly_df[hourly_df["sinav_donemi"] == exam_mode].copy()

    if len(df_full) < 1000:
        return None

    df_prophet = df_full.rename(
        columns={'datetime': 'ds', 'saatlik_ortalama_doluluk': 'y'}
    )[['ds', 'y', 'sinav_donemi']].sort_values(by='ds').copy()

    df_prophet['cap'] = capacity
    df_prophet['floor'] = 0

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        growth='logistic',
        seasonality_mode='additive',
        uncertainty_samples=100
    )

    model.add_regressor('sinav_donemi')

    try:
        model.fit(df_prophet)
    except Exception:
        return None

    future = model.make_future_dataframe(periods=168, freq='h', include_history=False)

    future['sinav_donemi'] = exam_mode
    future['cap'] = capacity
    future['floor'] = 0

    forecast = model.predict(future)

    forecast['yhat'] = forecast['yhat'].apply(lambda x: max(0, min(x, capacity)))
    forecast['yhat_lower'] = forecast['yhat_lower'].apply(lambda x: max(0, min(x, capacity)))
    forecast['yhat_upper'] = forecast['yhat_upper'].apply(lambda x: max(0, min(x, capacity)))

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


def forecast_for_slot(hourly_df, target_weekday, target_hour, exam_mode, capacity):
    """Belirli bir gün ve saat aralığı için 4 klasik modeli çalıştırır ve en iyisini seçer."""
    if exam_mode == 1:
        sub = hourly_df[hourly_df["sinav_donemi"] == 1].copy()
    else:
        sub = hourly_df[hourly_df["sinav_donemi"] == 0].copy()

    slot_sub = sub[(sub["weekday"] == target_weekday) & (sub["hour"] == target_hour)].copy()

    if slot_sub.empty or slot_sub["saatlik_ortalama_doluluk"].nunique() <= 1:
        raise ValueError("Bu gün/saat aralığı için yeterli ya da değişken veri yok.")

    y_values = slot_sub["saatlik_ortalama_doluluk"].reset_index(drop=True)

    results = {}

    ma_pred, ma_err = model_moving_average(y_values, window=10)
    results["Moving Average (MA)"] = (ma_pred, ma_err)
    es_pred, es_err = model_exponential_smoothing(y_values, alpha=0.35)
    results["Exponential Smoothing (ES)"] = (es_pred, es_err)
    hw_pred, hw_err = model_holt_winters_additive(y_values, alpha=0.3, beta=0.15, gamma=0.1, m=4)
    results["Holt-Winters (HW)"] = (hw_pred, hw_err)
    sd_pred, sd_err = model_seasonal_decomposition(y_values, m=4)
    results["Seasonal Decomposition (SD)"] = (sd_pred, sd_err)

    if not results:
        raise ValueError("Hiçbir model, bu veri seti üzerinde geçerli bir tahmin üretemedi.")

    candidate_models = [k for k, v in results.items() if not np.isnan(v[1])]

    if not candidate_models:
        raise ValueError("4 Klasik modelden geçerli sonuç alınamadı.")

    best_model = min(candidate_models, key=lambda k: results[k][1])
    best_pred, best_err = results[best_model]

    sigma = best_err
    interval_low = max(0, best_pred - 1.28 * sigma)
    interval_high = min(capacity, best_pred + 1.28 * sigma)
    best_pred = max(0, min(best_pred, capacity))

    return best_model, best_pred, best_err, interval_low, interval_high, results


# ======================
# 4. CUSTOMTKINTER GUI (CANLI VERİ ENTEGRASYONU)
# ======================

class LibTrackApp(ctk.CTk):
    def __init__(self, hourly_data, min_date, max_date, capacity):
        super().__init__()

        self.hourly = hourly_data
        self.min_date = min_date
        self.max_date = max_date
        self.capacity = capacity
        self.prophet_forecast = None

        self.title("📚 LibTrack AI - Kütüphane Doluluk Tahmini")
        self.geometry("1000x750")

        # Ana grid konfigürasyonu: Tek satır, kaydırılabilir çerçeve için
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.scrollable_main_frame = ctk.CTkScrollableFrame(self)
        self.scrollable_main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.scrollable_main_frame.grid_columnconfigure(0, weight=1)

        # Tüm eski bileşenler artık self.scrollable_main_frame içine grid edilecek.

        # --- 1. Başlık (row=0 of scroll frame) ---
        title_lbl = ctk.CTkLabel(self.scrollable_main_frame, text="Kütüphane Doluluk Tahmin Motoru",
                                 font=ctk.CTkFont(size=24, weight="bold"))
        title_lbl.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")  # Padding azaltıldı

        # --- 2. Canlı Veri Çerçevesi (row=1 of scroll frame) ---
        self.live_data_frame = ctk.CTkFrame(self.scrollable_main_frame, fg_color="transparent")
        self.live_data_frame.grid(row=1, column=0, padx=10, pady=(0, 5), sticky="ew")  # Padding azaltıldı
        self.live_data_frame.grid_columnconfigure(0, weight=1)
        self.live_data_frame.grid_columnconfigure(1, weight=0)

        # 1. Canlı Doluluk Kartı
        self.card_live = self.create_result_card(self.live_data_frame, "🔴 ANLIK DOLULUK (Canlı DB)", "DB Bağlanıyor...",
                                                 0, start_row=0)

        # 2. Canlı Veri Güncelle Butonu
        self.btn_live_update = ctk.CTkButton(
            self.live_data_frame,
            text="🔄 Canlı Veriyi Güncelle",
            command=self.update_live_occupancy,
            height=50,
            font=ctk.CTkFont(size=15, weight="bold")
        )
        self.btn_live_update.grid(row=0, column=1, padx=10, pady=5, sticky="e")
        # ---------------------------------------------

        # --- 3. Notebook Konfigürasyonu (row=2 of scroll frame) ---
        self.nb = ttk.Notebook(self.scrollable_main_frame)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background=self._apply_appearance_mode(ctk.ThemeManager.theme["CTk"]["fg_color"]))
        style.configure("TNotebook.Tab",
                        background=self._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"]),
                        padding=[10, 5])
        style.map("TNotebook.Tab", background=[("selected", ctk.ThemeManager.theme["CTkButton"]["fg_color"][1])])

        # Notebook, artık scrollable_main_frame içinde row=2'de
        self.nb.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")

        # Sekme Kurulumları
        self.f_slot = ctk.CTkFrame(self.nb)
        self.nb.add(self.f_slot, text=" 🎯 Tek Slot Tahmini ")
        self.setup_slot_prediction_tab(self.f_slot)

        self.f_prophet = ctk.CTkFrame(self.nb)
        self.nb.add(self.f_prophet, text=" 📊 Haftalık Genel Tahmin ")
        self.setup_prophet_tab(self.f_prophet)

        # İlk çalıştırmalar
        self.update_live_occupancy(initial_run=True)
        self.initial_prophet_run()
    def update_live_occupancy(self, initial_run=False):
        """DB'den canlı doluluk verisini çeker ve kartı günceller."""
        self.card_live.configure(text_color="gray", text="Yükleniyor...")
        self.update_idletasks()

        # Veri çekme
        latest_occupancy = fetch_latest_occupancy(DB_CONFIG)

        if isinstance(latest_occupancy, float) or isinstance(latest_occupancy, int):
            # Başarılı veri çekimi
            latest_occupancy = round(latest_occupancy)  # Kişi sayısını tam sayıya yuvarla
            perc = 100 * latest_occupancy / self.capacity

            # Kartı başarıyla güncelle
            self.card_live.configure(
                text_color="green",
                text=f"{latest_occupancy} Kişi ({perc:.1f}%)"
            )
            if not initial_run:
                messagebox.showinfo("Başarılı", f"Canlı doluluk verisi başarıyla çekildi: {latest_occupancy} kişi.")

        else:
            # Hata veya veri yok
            self.card_live.configure(text_color="red", text=str(latest_occupancy))
            if not initial_run:
                messagebox.showerror("Hata", f"Canlı veriye bağlanırken sorun oluştu:\n{latest_occupancy}")

    def create_result_card(self, parent, title, value, col, start_row=0):
        """Tekli sonuç kartı oluşturur (OKUNURLUK İÇİN YÜKSEK KONTRAST)."""
        card = ctk.CTkFrame(parent, fg_color="white")
        card.grid(row=start_row, column=col, padx=10, pady=10, sticky="nsew")

        title_lbl = ctk.CTkLabel(card, text=title, font=("Arial", 12))
        title_lbl.pack(padx=10, pady=(10, 0), anchor="w")

        value_lbl = ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=28, weight="bold"))
        value_lbl.pack(padx=10, pady=(0, 10), anchor="center")

        return value_lbl

    def setup_slot_prediction_tab(self, parent_frame):
        parent_frame.grid_columnconfigure(0, weight=1)
        parent_frame.grid_rowconfigure(3, weight=1)  # Detay alanı 3. satırda kalmaya devam ediyor

        # --- Giriş ve Kontrol Çerçevesi (row=0) ---
        control_frame = ctk.CTkFrame(parent_frame)
        control_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        control_frame.grid_columnconfigure((0, 2), weight=1)
        control_frame.grid_columnconfigure(1, weight=0)

        # Girişler... (Aynı)
        input_subframe = ctk.CTkFrame(control_frame, fg_color="transparent")
        input_subframe.grid(row=0, column=0, rowspan=2, padx=(10, 20), pady=10, sticky="nsw")
        input_subframe.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(input_subframe, text="1. Tahmin Tarihi:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        if HAS_TKCALENDAR:
            self.date_entry_frame = ctk.CTkFrame(input_subframe, fg_color="transparent", width=150)
            self.date_entry_frame.grid(row=0, column=1, padx=10, pady=5, sticky="w")
            self.date_entry = DateEntry(
                self.date_entry_frame, width=12, date_pattern="yyyy-mm-dd", mindate=self.min_date, font=("Arial", 12)
            )
            self.date_entry.set_date(self.max_date)
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

        # --- Tahmin Butonu ---
        self.btn = ctk.CTkButton(control_frame, text="🚀 Tek Slot Tahminini Çalıştır", command=self.make_slot_forecast,
                                 font=ctk.CTkFont(size=15, weight="bold"), height=50)
        self.btn.grid(row=0, column=1, rowspan=2, padx=20, pady=10, sticky="nsew")

        # --- Sonuç Kartları Çerçevesi (row=1) ---
        self.result_frame = ctk.CTkFrame(parent_frame, fg_color="transparent")
        self.result_frame.grid(row=1, column=0, padx=20, pady=(10, 10), sticky="ew")
        self.result_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.result_frame.grid_rowconfigure(1, weight=1)

        # Durum Göstergesi (row=0)
        self.status_lbl = ctk.CTkLabel(self.result_frame, text="Durum: Tahmin Yapılmadı",
                                       font=ctk.CTkFont(size=18, weight="bold"),
                                       anchor="w")
        self.status_lbl.grid(row=0, column=0, columnspan=3, padx=10, pady=(5, 5), sticky="w")

        # Tahmin, Yüzde ve Aralık Kartları (row=1)
        self.card_pred = self.create_result_card(self.result_frame, "👥 Tahmin (Kişi)", "N/A", 0, start_row=1)
        self.card_perc = self.create_result_card(self.result_frame, "📈 Doluluk Oranı", "N/A", 1, start_row=1)
        self.card_interval = self.create_result_card(self.result_frame, "🔒 Güven Aralığı (~80%)", "N/A", 2, start_row=1)

        # --- Detay ve Hata Alanı (row=2) ---
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

    def setup_prophet_tab(self, parent_frame):
        parent_frame.grid_columnconfigure(0, weight=1)
        parent_frame.grid_rowconfigure(3, weight=1)

        # Giriş Bölümü
        input_frame = ctk.CTkFrame(parent_frame)
        input_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")

        ctk.CTkLabel(input_frame, text="Prophet Haftalık Tahmin Ayarları:",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(padx=10, pady=5, anchor="w")

        self.prophet_exam_var = tk.IntVar(value=0)
        self.prophet_exam_check = ctk.CTkCheckBox(
            input_frame,
            text="Sınav Dönemi Verisi Kullan",
            variable=self.prophet_exam_var,
            onvalue=1, offvalue=0
        )
        self.prophet_exam_check.pack(padx=10, pady=5, anchor="w")

        self.btn_prophet = ctk.CTkButton(
            input_frame,
            text="♻️ Haftalık Tahmini Güncelle/Çalıştır",
            command=self.run_prophet_forecast,
            font=ctk.CTkFont(size=14, weight="bold"), height=30
        )
        self.btn_prophet.pack(padx=10, pady=10, fill="x")

        # --- Özet Çerçevesi ---
        self.summary_frame = ctk.CTkFrame(parent_frame, fg_color="transparent")
        self.summary_frame.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="ew")
        self.summary_frame.grid_columnconfigure((0, 1), weight=1)

        self.peak_day_lbl = ctk.CTkLabel(self.summary_frame, text="✨ Haftanın En Yoğun Günü: ?",
                                         font=ctk.CTkFont(size=14, weight="bold"), anchor="w", text_color="orange")
        self.peak_day_lbl.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.peak_hour_lbl = ctk.CTkLabel(self.summary_frame, text="🔥 Haftanın En Yoğun Saati: ?",
                                          font=ctk.CTkFont(size=14, weight="bold"), anchor="w", text_color="red")
        self.peak_hour_lbl.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        # Sonuç Bölümü
        result_frame = ctk.CTkFrame(parent_frame)
        result_frame.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="nsew")
        result_frame.grid_columnconfigure(0, weight=1)
        result_frame.grid_rowconfigure(0, weight=1)

        ctk.CTkLabel(result_frame, text="📊 Detaylı Saatlik Tahminler (07:00 - 23:00)",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(padx=10, pady=(10, 5), anchor="w")

        self.prophet_textbox = ctk.CTkTextbox(result_frame, height=450, font=("Courier New", 12))
        self.prophet_textbox.pack(fill='both', expand=True, padx=10, pady=10)
        self.prophet_textbox.insert("0.0",
                                    "Haftalık tahmin sonuçları burada, gün bazında gruplanmış olarak gösterilecektir...")
        self.prophet_textbox.configure(state="disabled")

    def initial_prophet_run(self):
        """Uygulama başladığında varsayılan modda Prophet'ı çalıştırır."""
        self.run_prophet_forecast(silent=True)

    def run_prophet_forecast(self, silent=False):
        """Haftalık Prophet tahminini yapar ve Textbox'ı günceller."""
        if not HAS_PROPHET:
            if not silent:
                messagebox.showerror("Hata", "Prophet kütüphanesi yüklü değil. Tahmin yapılamıyor.")
            return

        current_mode = self.prophet_exam_var.get()

        self.prophet_textbox.configure(state="normal")
        self.prophet_textbox.delete("1.0", "end")
        self.prophet_textbox.insert("0.0", "Lütfen bekleyin, model hesaplanıyor...")
        self.update_idletasks()

        try:
            self.prophet_forecast = model_prophet_weekly_forecast(
                self.hourly, current_mode, self.capacity
            )
        except Exception as e:
            self.prophet_textbox.delete("1.0", "end")
            self.prophet_textbox.insert("0.0", f"Model eğitimi sırasında hata oluştu: {e}")
            self.prophet_textbox.configure(state="disabled")
            if not silent:
                messagebox.showerror("Hata", "Prophet model eğitimi başarısız oldu.")
            return

        if self.prophet_forecast is None:
            self.prophet_textbox.delete("1.0", "end")
            self.prophet_textbox.insert("0.0", "Yeterli Veri/Prophet Hatası: Model eğitilemedi.")
            self.prophet_textbox.configure(state="disabled")
            if not silent:
                messagebox.showwarning("Uyarı",
                                       "Prophet için yeterli veri bulunamadı veya model eğitimi başarısız oldu.")
            return

        # 1. Saat Filtreleme (07:00 - 23:59 arası)
        filtered_forecast = self.prophet_forecast.copy()
        filtered_forecast['hour'] = filtered_forecast['ds'].dt.hour
        filtered_forecast = filtered_forecast[(filtered_forecast['hour'] >= 7) & (filtered_forecast['hour'] <= 23)]

        gun_adlari = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']

        # 2. Kreatif İyileştirme: Haftalık Özet
        if not filtered_forecast.empty:
            max_yhat_row = filtered_forecast.loc[filtered_forecast['yhat'].idxmax()]
            max_yhat_day = gun_adlari[max_yhat_row['ds'].weekday()]
            max_yhat_hour = max_yhat_row['ds'].strftime('%H:00')
            max_yhat_value = max_yhat_row['yhat']

            self.peak_day_lbl.configure(text=f"✨ Haftanın En Yoğun Günü: {max_yhat_day} ({max_yhat_value:.1f} Kişi)")
            self.peak_hour_lbl.configure(text=f"🔥 Haftanın En Yoğun Saati: {max_yhat_hour} ({max_yhat_value:.1f} Kişi)")
        else:
            self.peak_day_lbl.configure(text="✨ Haftanın En Yoğun Günü: Veri Yok")
            self.peak_hour_lbl.configure(text="🔥 Haftanın En Yoğun Saati: Veri Yok")

        # 3. Detaylı Yazdırma
        output = ""
        header = "SAAT | TAHMİN (Kişi) | % DOLULUK | GÜVEN ARALIĞI   | DURUM\n"
        separator = "-----+---------------+-----------+-----------------+---------------\n"

        for day, group in filtered_forecast.groupby(filtered_forecast['ds'].dt.date):
            day_name = gun_adlari[group['ds'].iloc[0].weekday()]

            output += f"=================================================================================\n"
            output += f"🗓️ {day.strftime('%Y-%m-%d')} - {day_name.upper()}\n"
            output += "=================================================================================\n"
            output += header
            output += separator

            for _, row in group.iterrows():
                hour = row['ds'].strftime('%H:%M')
                yhat = f"{row['yhat']:.1f}"
                yhat_lower = f"{row['yhat_lower']:.1f}"
                yhat_upper = f"{row['yhat_upper']:.1f}"
                perc = (row['yhat'] / self.capacity) * 100

                # Doluluk durumu etiketi
                durum = ""
                if perc > 80:
                    durum = "🔴 YÜKSEK"
                elif perc > 50:
                    durum = "🟡 ORTA"
                else:
                    durum = "🟢 DÜŞÜK"

                # Formatlama: String hizalama kullanıldı
                output += f"{hour:<4} | {yhat:^13} | {perc:^9.1f}% | {yhat_lower}-{yhat_upper:^14} | {durum}\n"

            output += "\n"

        self.prophet_textbox.delete("1.0", "end")
        self.prophet_textbox.insert("0.0", output)
        self.prophet_textbox.configure(state="disabled")

        if not silent:
            messagebox.showinfo("Başarılı", "Prophet Haftalık Tahmini Başarıyla Güncellendi.")

    def make_slot_forecast(self):
        """4 Klasik Model ile tek slot tahminini yapar."""
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")

        # Kartları sıfırla
        self.card_pred.configure(text="Hesaplanıyor...")
        self.card_perc.configure(text="Hesaplanıyor...")
        self.card_interval.configure(text="Hesaplanıyor...")
        self.status_lbl.configure(text="Durum: Hesaplanıyor...", text_color="gray")
        self.update_idletasks()

        try:
            # --- Giriş Verilerini Al ---
            if HAS_TKCALENDAR:
                date = pd.to_datetime(self.date_entry.get_date())
            else:
                date_str_in = self.date_entry.get().strip()
                if not date_str_in: raise ValueError("Lütfen geçerli bir tarih girin.")
                date = pd.to_datetime(date_str_in)

            date_str = date.strftime("%Y-%m-%d")
            weekday = date.weekday()
            slot_str = self.slot_combo.get().strip()
            if not slot_str: raise ValueError("Lütfen bir saat slotu seçin.")
            start_hour = int(slot_str.split(":")[0])
            exam_mode = self.exam_var.get()

            # --- Tahmin Fonksiyonunu Çağır ---
            best_model, best_pred, best_err, low, high, all_results = forecast_for_slot(
                self.hourly, weekday, start_hour, exam_mode, self.capacity
            )

            # Hesaplamalar
            perc = 100 * best_pred / self.capacity
            perc_low = 100 * low / self.capacity
            perc_high = 100 * high / self.capacity
            gun_adlari = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']

            # Durum Tespiti
            status = ""
            status_color = ""
            if perc > 80:
                status = "YÜKSEK (Yoğunluk bekleniyor)"
                status_color = "red"
            elif perc > 50:
                status = "ORTA (Yer bulmak zorlaşabilir)"
                status_color = "orange"
            else:
                status = "DÜŞÜK (Rahat çalışma ortamı)"
                status_color = "green"

            # --- Kartları Güncelle ---
            self.card_pred.configure(text=f"{best_pred:.1f} Kişi")
            self.card_perc.configure(text=f"{perc:.1f}%")
            self.card_interval.configure(text=f"[{low:.1f} - {high:.1f}]")

            # --- Status Label'ı Güncelle ---
            self.status_lbl.configure(text=f"Durum: {status}", text_color=status_color)

            # --- Detay Metni Oluştur ---
            output = f"📅 Tarih: {date_str} ({gun_adlari[weekday]})\n"
            output += f"⏰ Slot: {slot_str}\n"
            output += f"📚 Dönem: {'Sınav Dönemi Verisi' if exam_mode == 1 else 'Normal Dönem Verisi'}\n"
            output += "--------------------------------------------------------\n"
            output += f"✨ En İyi Model: {best_model}\n"
            output += f"🎯 Model Hata Payı (MAE): {best_err:.2f}\n\n"
            output += f"⚖️ KESİN ARALIK DETAYI:\n"
            output += f"   - % Aralığı: {perc_low:.1f}% - {perc_high:.1f}%\n\n"
            output += "📊 Tüm Modellerin Hata Değerleri (MAE):\n"
            for name, (pred, err) in all_results.items():
                suffix = "  <-- Seçilen Model" if name == best_model else ""
                output += f"  - {name}: {err:.2f}{suffix}\n"

            self.result_text.insert(tk.END, output)

        except ValueError as e:
            msg = str(e)
            messagebox.showerror("Giriş Hatası", msg)
            self.card_pred.configure(text="HATA!")
            self.card_perc.configure(text="HATA!")
            self.card_interval.configure(text="HATA!")
            self.status_lbl.configure(text="Durum: HATA!", text_color="red")
            self.result_text.insert(tk.END, f"Hata: {msg}")
        except Exception as e:
            msg = f"Beklenmeyen bir hata oluştu: {str(e)}"
            messagebox.showerror("Tahmin Hatası", msg)
            self.card_pred.configure(text="HATA!")
            self.card_perc.configure(text="HATA!")
            self.card_interval.configure(text="HATA!")
            self.status_lbl.configure(text="Durum: HATA!", text_color="red")
            self.result_text.insert(tk.END, f"Hata: {msg}")
        finally:
            self.result_text.configure(state="disabled")


# ======================
# 5. ÇALIŞTIR
# ======================

if __name__ == "__main__":
    app = LibTrackApp(hourly, min_date, max_date, CAPACITY)
    app.mainloop()