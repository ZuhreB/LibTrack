import pandas as pd
import numpy as np

# Prophet Kontrolü
try:
    from prophet import Prophet

    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False


class ForecastingEngine:
    """
    Tüm istatistiksel ve makine öğrenmesi modellerini barındırır.
    """

    def __init__(self, capacity):
        self.capacity = capacity

    @staticmethod
    def mae(actual, predicted):
        actual = pd.Series(actual)
        predicted = pd.Series(predicted, index=actual.index)
        return (actual - predicted).abs().mean()

    # --- Klasik Modeller (Helper Functions) ---
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
        y = y.copy().reset_index(drop=True)
        n = len(y)
        if n < 2 * m: return self.model_exponential_smoothing(y, alpha=alpha)

        L0 = y.iloc[:m].mean()
        T0 = (y.iloc[m:2 * m].mean() - y.iloc[:m].mean()) / m if n >= 2 * m else (
                    y.iloc[1] - y.iloc[0]) if n > 1 else 0.0
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
        err = self.mae(y.loc[valid_idx], fitted.loc[valid_idx])
        pred_next = L[-1] + T[-1] + S[(n - m) % m]
        return pred_next, err

    def model_seasonal_decomposition(self, y, m=4):
        y = y.copy().reset_index(drop=True)
        n = len(y)
        if n < 2 * m: return self.model_exponential_smoothing(y, alpha=0.3)

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
            seasonal[i] = seasonal[i] / counts[i] if counts[i] > 0 else 0.0

        seasonal_series = pd.Series([seasonal[i % m] for i in range(n)], index=y.index)
        recon = trend + seasonal_series
        valid_mask = ~trend.isna()
        if valid_mask.sum() < 3: return self.model_exponential_smoothing(y, alpha=0.3)

        err = self.mae(y[valid_mask], recon[valid_mask])
        last_trend = trend[valid_mask].iloc[-1]
        pred_next = last_trend + seasonal[n % m]
        return pred_next, err

    # --- ANA TAHMİN FONKSİYONLARI ---

    def run_best_slot_forecast(self, hourly_df, target_weekday, target_hour, exam_mode):
        """Tek bir slot için en iyi modeli seçer."""
        sub = hourly_df[hourly_df["sinav_donemi"] == (1 if exam_mode == 1 else 0)].copy()
        slot_sub = sub[(sub["weekday"] == target_weekday) & (sub["hour"] == target_hour)].copy()

        if slot_sub.empty or slot_sub["saatlik_ortalama_doluluk"].nunique() <= 1:
            raise ValueError("Bu gün/saat aralığı için yeterli veri yok.")

        y_values = slot_sub["saatlik_ortalama_doluluk"].reset_index(drop=True)
        results = {}

        # 4 Modeli Yarıştır
        ma_pred, ma_err = self.model_moving_average(y_values, window=10)
        results["Moving Average (MA)"] = (ma_pred, ma_err)

        es_pred, es_err = self.model_exponential_smoothing(y_values, alpha=0.35)
        results["Exponential Smoothing (ES)"] = (es_pred, es_err)

        hw_pred, hw_err = self.model_holt_winters_additive(y_values, alpha=0.3, beta=0.15, gamma=0.1, m=4)
        results["Holt-Winters (HW)"] = (hw_pred, hw_err)

        sd_pred, sd_err = self.model_seasonal_decomposition(y_values, m=4)
        results["Seasonal Decomposition (SD)"] = (sd_pred, sd_err)

        # En iyisini seç
        candidate_models = [k for k, v in results.items() if not np.isnan(v[1])]
        if not candidate_models: raise ValueError("Modellerden sonuç alınamadı.")

        best_model = min(candidate_models, key=lambda k: results[k][1])
        best_pred, best_err = results[best_model]

        # Güven aralığı
        sigma = best_err
        interval_low = max(0, best_pred - 1.96 * sigma)
        interval_high = min(self.capacity, best_pred + 1.96 * sigma)
        best_pred = max(0, min(best_pred, self.capacity))

        return best_model, best_pred, best_err, interval_low, interval_high, results

    def run_prophet_weekly(self, hourly_df, exam_mode):
        """Prophet ile haftalık tahmin yapar."""
        if not HAS_PROPHET: return None

        df_full = hourly_df[hourly_df["sinav_donemi"] == exam_mode].copy()
        if len(df_full) < 1000: return None

        df_prophet = df_full.rename(columns={'datetime': 'ds', 'saatlik_ortalama_doluluk': 'y'})[
            ['ds', 'y', 'sinav_donemi']].sort_values(by='ds').copy()
        df_prophet['cap'] = self.capacity;
        df_prophet['floor'] = 0

        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True, growth='logistic',
                        seasonality_mode='additive', interval_width=0.95)
        model.add_regressor('sinav_donemi')

        try:
            model.fit(df_prophet)
        except:
            return None

        future = model.make_future_dataframe(periods=168, freq='h', include_history=False)
        future['sinav_donemi'] = exam_mode;
        future['cap'] = self.capacity;
        future['floor'] = 0

        forecast = model.predict(future)
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            forecast[col] = forecast[col].apply(lambda x: max(0, min(x, self.capacity)))

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]