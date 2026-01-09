import pandas as pd
import numpy as np

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False


class ForecastingEngine:
    def __init__(self, capacity):
        self.capacity = capacity

    @staticmethod
    def mae(actual, predicted):
        actual = pd.Series(actual)
        predicted = pd.Series(predicted, index=actual.index)
        return (actual - predicted).abs().mean()

    def model_moving_average(self, y, window=10):
        y = y.copy()
        if len(y) <= window:
            pred = y.mean()
            preds = pd.Series(pred, index=y.index)
            err = self.mae(y, preds)
        else:
            preds = y.rolling(window=window).mean()
            valid = preds.dropna()
            err = self.mae(y.loc[valid.index], valid)
            pred = valid.iloc[-1]
        return pred, err

    def model_exponential_smoothing(self, y, alpha=0.3):
        y = y.copy().reset_index(drop=True)
        s = [y.iloc[0]]
        preds = [y.iloc[0]]
        for i in range(1, len(y)):
            s.append(alpha * y.iloc[i] + (1 - alpha) * s[i - 1])
            preds.append(s[i - 1])

        s = pd.Series(s, index=y.index)
        preds = pd.Series(preds, index=y.index)
        err = self.mae(y, preds)

        pred_next = s.iloc[-1]
        return pred_next, err

    def model_holt_winters_additive(self, y, alpha=0.3, beta=0.1, gamma=0.1, m=4):
        # Holt-Winters Toplamsal Yöntemi (HW).
        # Level (L), Trend (T) ve Mevsimsellik (S) olmak üzere 3 bileşeni aynı anda takip ediyor.
        # m: Mevsimsel periyot uzunluğu (Bizim haftalık doluluk verimizde genellikle 7'dir, burada 4 olarak sabitlenmiş).
        y = y.copy().reset_index(drop=True)
        n = len(y)
        if n < 2 * m: return self.model_exponential_smoothing(y, alpha=alpha)  # Yeterli veri yoksa ES'ye dön

        # Başlangıç değerlerini (L0, T0, S) hesaplıyoruz.
        L0 = y.iloc[:m].mean()
        T0 = (y.iloc[m:2 * m].mean() - y.iloc[:m].mean()) / m if n >= 2 * m else (
                y.iloc[1] - y.iloc[0]) if n > 1 else 0.0
        S = [y.iloc[i] - L0 for i in range(m)]
        L, T, fitted = [L0], [T0], [L0 + T0 + S[0]]

        for t in range(1, n):
            # Mevsimsel indexi hesapla: (t - m) % m, ilk periyot için (t % m) kullanılır.
            idx_season = (t - m) % m
            Stm = S[idx_season] if t - m >= 0 else S[t % m]  # S'den uygun mevsimsellik bileşenini çek.

            # Level, Trend ve Mevsimsellik (L, T, S) bileşenlerini alpha, beta, gamma ile güncelle.
            Lt = alpha * (y.iloc[t] - Stm) + (1 - alpha) * (L[t - 1] + T[t - 1])
            Tt = beta * (Lt - L[t - 1]) + (1 - beta) * T[t - 1]
            St = gamma * (y.iloc[t] - Lt) + (1 - gamma) * Stm

            L.append(Lt);
            T.append(Tt);
            S[t % m] = St  # S listesi sabit boyutta (m) tutulur, her adımda sadece uygun index güncellenir.
            fitted.append(L[t - 1] + T[t - 1] + Stm)

        fitted = pd.Series(fitted, index=y.index)
        valid_idx = y.index[m:]
        err = self.mae(y.loc[valid_idx], fitted.loc[valid_idx])
        pred_next = L[-1] + T[-1] + (S[n % m])
        return pred_next, err

    def model_seasonal_decomposition(self, y, m=4):
        # Mevsimsel Ayrıştırma (Seasonal Decomposition - SD).
        # Trendi hareketli ortalama ile bul, Trendi veriden çıkar, kalanları (mevsimsellik) ortalamasını al.
        y = y.copy().reset_index(drop=True)
        n = len(y)
        if n < 2 * m: return self.model_exponential_smoothing(y, alpha=0.3)  # Yeterli veri yoksa ES'ye dön

        # 1. Trend Hesaplama: m boyutunda hareketli ortalama al.
        trend = y.rolling(window=m, center=True).mean()
        # 2. Mevsimsellik + Gürültü: Veriden trendi çıkar.
        detrended = y - trend

        seasonal = np.zeros(m);
        counts = np.zeros(m)
        # 3. Mevsimselliği Ayıkla: Aynı periyot indexine (i % m) düşen tüm detrended değerlerini topla.
        for i in range(n):
            idx = i % m
            if not np.isnan(detrended.iloc[i]):
                seasonal[idx] += detrended.iloc[i];
                counts[idx] += 1
        # 4. Mevsimsel Bileşenin Ortalama Etkisini Bul: Toplamı sayaca böl.
        for i in range(m):
            seasonal[i] = seasonal[i] / counts[i] if counts[i] > 0 else 0.0

        # Hesaplanan mevsimselliği seriye dönüştür.
        seasonal_series = pd.Series([seasonal[i % m] for i in range(n)], index=y.index)
        recon = trend + seasonal_series  # Yeniden oluşturulmuş seri (Trend + Mevsimsellik)
        valid_mask = ~trend.isna()
        if valid_mask.sum() < 3: return self.model_exponential_smoothing(y, alpha=0.3)

        # Hata hesaplaması: Gerçek veri ile yeniden oluşturulmuş seri arasındaki hata.
        err = self.mae(y[valid_mask], recon[valid_mask])
        last_trend = trend[valid_mask].iloc[-1]

        # Gelecek Tahmini: Son trend + Bir sonraki adıma denk gelen mevsimsel etki (n % m).
        # Buradaki n % m indeksleme, bir sonraki adımı doğru gösterir.
        pred_next = last_trend + seasonal[n % m]
        return pred_next, err

    def run_best_slot_forecast(self, hourly_df, target_weekday, target_hour, exam_mode):
        """Tek bir slot (belirli gün, belirli saat) için 4 modelin yarıştığı asıl fonksiyon."""
        # 1. Filtreleme: Önce Sınav Dönemine bak (0 veya 1).
        sub = hourly_df[hourly_df["sinav_donemi"] == (1 if exam_mode == 1 else 0)].copy()
        # Sonra tam o gün ve saat aralığına ait geçmiş verileri çek.
        slot_sub = sub[(sub["weekday"] == target_weekday) & (sub["hour"] == target_hour)].copy()

        if slot_sub.empty or slot_sub["saatlik_ortalama_doluluk"].nunique() <= 1:
            raise ValueError("Bu gün/saat aralığı için yeterli veri yok.")

        y_values = slot_sub["saatlik_ortalama_doluluk"].reset_index(drop=True)
        results = {}

        # 2. Modelleri Çalıştır ve Hata Skorlarını Al
        # Her model çalışır ve tahmin (pred) ile hata (err) değerini döndürür.

        # Moving Average
        ma_pred, ma_err = self.model_moving_average(y_values, window=10)
        results["Moving Average (MA)"] = (ma_pred, ma_err)

        # Exponential Smoothing
        es_pred, es_err = self.model_exponential_smoothing(y_values, alpha=0.35)
        results["Exponential Smoothing (ES)"] = (es_pred, es_err)

        # Holt-Winters (period=4, bu muhtemelen haftalık veri için yanlış ama kodda böyle)
        hw_pred, hw_err = self.model_holt_winters_additive(y_values, alpha=0.3, beta=0.15, gamma=0.1, m=4)
        results["Holt-Winters (HW)"] = (hw_pred, hw_err)

        # Seasonal Decomposition (period=4, aynı şekilde)
        sd_pred, sd_err = self.model_seasonal_decomposition(y_values, m=4)
        results["Seasonal Decomposition (SD)"] = (sd_pred, sd_err)

        # 3. En İyisini Seç
        candidate_models = [k for k, v in results.items() if not np.isnan(v[1])]
        if not candidate_models: raise ValueError("Modellerden geçerli sonuç alınamadı.")

        # Hata değeri (MAE) en düşük olanı (min) bul.
        best_model = min(candidate_models, key=lambda k: results[k][1])
        best_pred, best_err = results[best_model]

        # 4. Güven Aralığı Hesaplama ve Sınırlandırma
        # Tahminimiz ne kadar güvenilir? (1.96 * Sigma = %95 Güven Aralığı)
        sigma = best_err  # Hatayı (MAE) standart sapma (sigma) yerine kullanıyoruz.
        interval_low = max(0, best_pred - 1.96 * sigma)
        interval_high = min(self.capacity, best_pred + 1.96 * sigma)

        # Tahmin değerini de 0 ile kapasite arasına zorla (doluluk eksi veya kapasiteden fazla olamaz).
        best_pred = max(0, min(best_pred, self.capacity))

        return best_model, best_pred, best_err, interval_low, interval_high, results

    def run_prophet_weekly(self, hourly_df, exam_mode, target_start_date=None):

        if not HAS_PROPHET: return None

        df_full = hourly_df[hourly_df["sinav_donemi"] == exam_mode].copy()
        if len(df_full) < 100: return None

        df_prophet = df_full.rename(columns={'datetime': 'ds', 'saatlik_ortalama_doluluk': 'y'})[
            ['ds', 'y', 'sinav_donemi']].sort_values(by='ds').copy()
        df_prophet['cap'] = self.capacity
        df_prophet['floor'] = 0

        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True, growth='logistic',
                        seasonality_mode='additive', interval_width=0.95)
        model.add_regressor('sinav_donemi')

        try:
            model.fit(df_prophet)


            if target_start_date:
                future_dates = pd.date_range(start=target_start_date, periods=168, freq='h')
                future = pd.DataFrame({'ds': future_dates})
            else:
                future = model.make_future_dataframe(periods=168, freq='h', include_history=False)

            future['sinav_donemi'] = exam_mode
            future['cap'] = self.capacity
            future['floor'] = 0

            forecast = model.predict(future)

            for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                forecast[col] = forecast[col].apply(lambda x: max(0, min(x, self.capacity)))

            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        except Exception as e:
            print(f"Prophet Hatası: {e}")
            return None