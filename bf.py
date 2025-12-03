import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk  # CustomTkinter'ı import ediyoruz

try:
    from tkcalendar import DateEntry

    HAS_TKCALENDAR = True
except ImportError:
    HAS_TKCALENDAR = False

# CustomTkinter ayarları
ctk.set_appearance_mode("System")  # System, Dark, Light
ctk.set_default_color_theme("blue")  # themes: blue, dark-blue, green

# ======================
# 1. VERİYİ YÜKLE / HAZIRLA
# ======================

# Not: Bu kısım orijinal koddan korunmuştur.
CSV_PATH = "libtrack_dataset_bounded_realistic_v2.csv"  # Kendi dosya adın buysa dokunma

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    messagebox.showerror("Hata", f"{CSV_PATH} dosyası bulunamadı. Lütfen dosyanın doğru yolda olduğundan emin olun.")
    exit()

df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])

# Sadece saatlik ortalama olan satırlar
hourly = df[df["saatlik_ortalama_doluluk"].notnull()].copy()
hourly["saatlik_ortalama_doluluk"] = hourly["saatlik_ortalama_doluluk"].astype(float)
hourly["date"] = pd.to_datetime(hourly["date"])
hourly["hour"] = hourly["datetime"].dt.hour
hourly["weekday"] = hourly["datetime"].dt.weekday  # 0=Mon ... 6=Sun

min_date = hourly["date"].min().date()
max_date = hourly["date"].max().date()

CAPACITY = 432  # Kütüphane kapasitesi


# ======================
# 2. FORECAST MODELLERİ
# ======================

def mae(actual, predicted):
    actual = pd.Series(actual)
    predicted = pd.Series(predicted, index=actual.index)
    return (actual - predicted).abs().mean()


# ---- Moving Average (window=10) ----
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


# ---- Simple Exponential Smoothing ----
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


# ---- Holt-Winters (additive triple exponential smoothing) ----
def model_holt_winters_additive(y, alpha=0.3, beta=0.1, gamma=0.1, m=4):
    """
    Basit additive Holt-Winters (tek adım ileri tahmin için).
    m: season length (örneğin 4 ~ yaklaşık “4 haftalık” pattern gibi).
    """
    y = y.copy().reset_index(drop=True)
    n = len(y)

    # Veri çok azsa ES'e fallback
    if n < 2 * m:
        return model_exponential_smoothing(y, alpha=alpha)

    # Başlangıç level
    L0 = y.iloc[:m].mean()

    # Başlangıç trend
    if n >= 2 * m:
        T0 = (y.iloc[m:2 * m].mean() - y.iloc[:m].mean()) / m
    else:
        T0 = (y.iloc[1] - y.iloc[0]) if n > 1 else 0.0

    # Başlangıç seasonal bileşenler
    S = [y.iloc[i] - L0 for i in range(m)]

    L = [L0]
    T = [T0]
    fitted = [L0 + T0 + S[0]]  # t=0 tahmini (kabaca)

    for t in range(1, n):
        idx_season = (t - m) % m
        if t - m >= 0:
            Stm = S[idx_season]
        else:
            Stm = S[t % m]

        Lt = alpha * (y.iloc[t] - Stm) + (1 - alpha) * (L[t - 1] + T[t - 1])
        Tt = beta * (Lt - L[t - 1]) + (1 - beta) * T[t - 1]
        St = gamma * (y.iloc[t] - Lt) + (1 - gamma) * Stm

        L.append(Lt)
        T.append(Tt)
        S[t % m] = St

        fitted.append(L[t - 1] + T[t - 1] + Stm)

    fitted = pd.Series(fitted, index=y.index)
    valid_idx = y.index[m:]
    err = mae(y.loc[valid_idx], fitted.loc[valid_idx])

    # ileri 1 adım tahmin
    next_season_index = (n - m) % m
    St_future = S[next_season_index]
    pred_next = L[-1] + T[-1] + St_future
    return pred_next, err


# ---- Seasonal Decomposition (manuel additive) ----
def model_seasonal_decomposition(y, m=4):
    """
    Basit additive seasonal decomposition:
    y = trend + seasonal + noise
    Trend: moving average (window=m)
    Seasonal: m uzunluklu pattern (ortalama residual'a göre)
    """
    y = y.copy().reset_index(drop=True)
    n = len(y)

    if n < 2 * m:
        # Veri azsa yine ES'e fallback
        return model_exponential_smoothing(y, alpha=0.3)

    # Trend (centered moving average)
    trend = y.rolling(window=m, center=True).mean()

    # Detrended
    detrended = y - trend

    # Mevsimsel bileşenleri hesapla
    seasonal = np.zeros(m)
    counts = np.zeros(m)

    for i in range(n):
        idx = i % m
        if not np.isnan(detrended.iloc[i]):
            seasonal[idx] += detrended.iloc[i]
            counts[idx] += 1

    for i in range(m):
        if counts[i] > 0:
            seasonal[i] /= counts[i]
        else:
            seasonal[i] = 0.0

    # Rekonstrüksiyon
    seasonal_series = pd.Series([seasonal[i % m] for i in range(n)], index=y.index)
    recon = trend + seasonal_series

    # Hata: yalnızca trend'in NaN olmadığı kısımlar
    valid_mask = ~trend.isna()
    if valid_mask.sum() < 3:
        return model_exponential_smoothing(y, alpha=0.3)

    err = mae(y[valid_mask], recon[valid_mask])

    # İleri 1 adım tahmin
    last_trend = trend[valid_mask].iloc[-1]
    next_season_idx = n % m
    pred_next = last_trend + seasonal[next_season_idx]

    return pred_next, err


# ======================
# 3. SLOT BAZLI FORECAST (Orijinal koddan kopyalanmıştır)
# ======================

def forecast_for_slot(hourly_df, target_weekday, target_hour, exam_mode):
    """
    target_weekday: 0=Mon,...,6=Sun
    target_hour: slot başlangıcı (9 => 09:00-10:00)
    exam_mode: 0 -> normal dönem, 1 -> sadece sınav dönemi verisi
    """
    # Sınav / normal filtre
    if exam_mode == 1:
        sub = hourly_df[hourly_df["sinav_donemi"] == 1].copy()
    else:
        sub = hourly_df[hourly_df["sinav_donemi"] == 0].copy()

    # Hafta günü + saat filtresi
    sub = sub[(sub["weekday"] == target_weekday) & (sub["hour"] == target_hour)].copy()

    if sub.empty or sub["saatlik_ortalama_doluluk"].nunique() <= 1:
        raise ValueError("Bu gün/saat aralığı için yeterli ya da değişken veri yok.")

    y = sub["saatlik_ortalama_doluluk"].reset_index(drop=True)

    results = {}

    # 1) Moving Average
    ma_pred, ma_err = model_moving_average(y, window=10)
    results["Moving Average"] = (ma_pred, ma_err)

    # 2) Exponential Smoothing
    es_pred, es_err = model_exponential_smoothing(y, alpha=0.35)
    results["Exponential Smoothing"] = (es_pred, es_err)

    # 3) Holt-Winters
    hw_pred, hw_err = model_holt_winters_additive(y, alpha=0.3, beta=0.15, gamma=0.1, m=4)
    results["Holt-Winters"] = (hw_pred, hw_err)

    # 4) Seasonal Decomposition
    sd_pred, sd_err = model_seasonal_decomposition(y, m=4)
    results["Seasonal Decomposition"] = (sd_pred, sd_err)

    # En iyi modeli seç
    candidate_models = list(results.keys())
    best_model = min(candidate_models, key=lambda k: results[k][1])
    best_pred, best_err = results[best_model]

    # ~80% interval için sigma ~ best_err alıp ±1.28*sigma
    sigma = best_err
    interval_low = best_pred - 1.28 * sigma
    interval_high = best_pred + 1.28 * sigma

    return best_model, best_pred, best_err, interval_low, interval_high, results


# ======================
# 4. CUSTOMTKINTER GUI
# ======================

class LibTrackApp(ctk.CTk):
    def __init__(self, hourly_data, min_date, max_date, capacity):
        super().__init__()

        self.hourly = hourly_data
        self.min_date = min_date
        self.max_date = max_date
        self.capacity = capacity

        # --- Ana Ayarlar ---
        self.title("📚 LibTrack AI - Kütüphane Doluluk Tahmini")
        self.geometry("750x650")  # Daha büyük bir pencere
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)  # Sonuç alanı için genişleme

        # --- Başlık ---
        title_lbl = ctk.CTkLabel(self, text="Kütüphane Doluluk Tahmin Motoru",
                                 font=ctk.CTkFont(size=20, weight="bold"))
        title_lbl.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        # --- Giriş Parametreleri Çerçevesi ---
        input_frame = ctk.CTkFrame(self)
        input_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        input_frame.grid_columnconfigure(1, weight=1)

        # 1. Tarih Seçimi
        ctk.CTkLabel(input_frame, text="Tahmin Tarihi:").grid(row=0, column=0, padx=10, pady=10, sticky="w")

        # tkcalendar kullanma
        if HAS_TKCALENDAR:
            self.date_entry_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
            self.date_entry_frame.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
            self.date_entry = DateEntry(
                self.date_entry_frame,
                width=12,
                date_pattern="yyyy-mm-dd",
                mindate=self.min_date,
                font=("Arial", 12)
            )
            self.date_entry.set_date(self.max_date)
            # CustomTkinter stilini taklit etmek için küçük bir çerçeve
            self.date_entry.pack(fill='x', expand=True, padx=2)
        else:
            self.date_entry = ctk.CTkEntry(input_frame, width=150)
            self.date_entry.insert(0, "2025-12-25")
            self.date_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        # 2. Saat Slotu Seçimi
        ctk.CTkLabel(input_frame, text="Saat Slotu:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        slots = [f"{h:02d}:00-{h + 1:02d}:00" for h in range(8, 23)]
        self.slot_combo = ctk.CTkComboBox(input_frame, values=slots, width=150, state="readonly")
        self.slot_combo.set(slots[4])  # 12:00-13:00
        self.slot_combo.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        # 3. Sınav Dönemi Seçimi
        self.exam_var = tk.IntVar(value=0)
        self.exam_check = ctk.CTkCheckBox(
            input_frame,
            text="Sınav Dönemi Deseni Kullan (Exam Mode)",
            variable=self.exam_var,
            onvalue=1, offvalue=0
        )
        self.exam_check.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="w")

        # --- Tahmin Butonu ---
        self.btn = ctk.CTkButton(self, text="🚀 Tahmini Çalıştır", command=self.make_forecast,
                                 font=ctk.CTkFont(size=14, weight="bold"), height=40)
        self.btn.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

        # --- Sonuç Alanı ---
        self.result_frame = ctk.CTkFrame(self)
        self.result_frame.grid(row=4, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.result_frame.grid_columnconfigure(0, weight=1)
        self.result_frame.grid_rowconfigure(0, weight=1)

        ctk.CTkLabel(self.result_frame, text="✅ Sonuçlar:",
                     font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")

        self.result_text = ctk.CTkTextbox(self.result_frame, height=250, width=650)
        self.result_text.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="nsew")
        self.result_text.insert("0.0", "Tahmin sonuçları burada görünecektir...")
        self.result_text.configure(state="disabled")

    def make_forecast(self):
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")

        try:
            # Tarih
            if HAS_TKCALENDAR:
                # tkcalendar DateEntry'den tarih alma
                date = pd.to_datetime(self.date_entry.get_date())
            else:
                date_str_in = self.date_entry.get().strip()
                if not date_str_in:
                    raise ValueError("Lütfen bir tarih girin (YYYY-MM-DD).")
                date = pd.to_datetime(date_str_in)

            date_str = date.strftime("%Y-%m-%d")
            weekday = date.weekday()  # 0=Pzt, 6=Paz

            # Slot
            slot_str = self.slot_combo.get().strip()
            if not slot_str:
                raise ValueError("Lütfen bir saat slotu seçin.")

            start_hour = int(slot_str.split(":")[0])
            exam_mode = self.exam_var.get()

            # Tahmin Fonksiyonunu Çağır
            best_model, best_pred, best_err, low, high, all_results = forecast_for_slot(
                self.hourly, weekday, start_hour, exam_mode
            )

            # Hesaplamalar
            perc = 100 * best_pred / self.capacity
            perc_low = 100 * low / self.capacity
            perc_high = 100 * high / self.capacity

            # --- Sonuç Metni Oluşturma ---
            output = f"📅 Seçilen Tarih: {date_str} ({['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar'][weekday]})\n"
            output += f"⏰ Saat Slotu: {slot_str}\n"
            output += f"📚 Dönem Modu: {'Sınav Dönemi Verisi' if exam_mode == 1 else 'Normal Dönem Verisi'}\n"
            output += "--------------------------------------------------------\n\n"

            # Ana Tahmin
            output += f"✨ En İyi Model: {best_model} (MAE = {best_err:.2f})\n"
            output += f"👥 **Tahmini Doluluk:** **{best_pred:.1f}** kişi (Kapasitenin ~**{perc:.1f}%**'si)\n"
            output += f"🔒 Güven Aralığı (~80%): **[{low:.1f} - {high:.1f}]** kişi\n"
            output += f"   (% aralığı: {perc_low:.1f}% - {perc_high:.1f}%)\n\n"

            # Kapasite Durumu
            status = ""
            if perc > 80:
                status = "🔴 YÜKSEK: Kapasite sınırına yakın, yoğunluk bekleniyor."
            elif perc > 50:
                status = "🟡 ORTA: Yoğunluk artabilir, rahat bir yer bulmak zorlaşabilir."
            else:
                status = "🟢 DÜŞÜK: Genel olarak boş yer bulunabilir, rahat çalışma ortamı."

            output += f"ℹ️ **Durum Değerlendirmesi:** {status}\n\n"

            # Tüm Modeller
            output += "📊 Diğer Modellerin Hata Değerleri (MAE):\n"
            for name, (pred, err) in all_results.items():
                suffix = "  <-- Seçilen Model" if name == best_model else ""
                output += f"  - {name}: {err:.2f}{suffix}\n"

            self.result_text.insert(tk.END, output)

        except ValueError as e:
            messagebox.showerror("Giriş Hatası", str(e))
            self.result_text.insert(tk.END, f"Hata: {str(e)}")
        except Exception as e:
            messagebox.showerror("Tahmin Hatası", f"Beklenmeyen bir hata oluştu: {str(e)}")
            self.result_text.insert(tk.END, f"Beklenmeyen Hata: {str(e)}")
        finally:
            self.result_text.configure(state="disabled")


# ======================
# 5. ÇALIŞTIR
# ======================

if __name__ == "__main__":
    app = LibTrackApp(hourly, min_date, max_date, CAPACITY)
    app.mainloop()