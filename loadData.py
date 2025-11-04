import tkinter as tk
from tkinter import ttk
import pandas as pd
from prophet import Prophet
import time
import random
from datetime import datetime

data = {
    'zaman_damgasi': ['2025-01-06 14:00:00', '2025-01-07 14:00:00', '2025-01-08 14:00:00',
                      '2025-01-09 14:00:00', '2025-01-10 14:00:00', '2025-01-11 14:00:00',
                      '2025-01-12 14:00:00', '2025-01-13 14:00:00'],
    'doluluk_yuzdesi': [55, 62, 65, 90, 88, 98, 95, 58],
    'sinav_donemi': [0, 0, 0, 1, 1, 1, 1, 0]
}

df = pd.DataFrame(data)
df = df.rename(columns={'zaman_damgasi': 'ds', 'doluluk_yuzdesi': 'y'})
df['ds'] = pd.to_datetime(df['ds'])

m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True)
m.add_regressor('sinav_donemi')
m.fit(df)

# Gelecek 4 günlük tahmin
future = m.make_future_dataframe(periods=4, freq='D')
gelecek_sinav_donemi = [0, 0, 1, 1]
future['sinav_donemi'] = df['sinav_donemi']
future.loc[future.index[-4:], 'sinav_donemi'] = gelecek_sinav_donemi

forecast = m.predict(future)
tahminler_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(4)


def format_prediction_data(tahminler_df, future_df):
    """Prophet çıktısını GUI'de gösterilecek formata dönüştürür."""
    formatted_data = []
    future_tail = future_df.tail(4).reset_index(drop=True)

    for i, row in enumerate(tahminler_df.itertuples()):
        tarih = row.ds.strftime('%Y-%m-%d')
        tahmin_yuzdesi = round(row.yhat, 0)
        guven_araligi_yari = round((row.yhat_upper - row.yhat_lower) / 2, 0)

        # future_tail'in sıfırdan başlayan indeksi (i) kullanıldı.
        sinav_donemi = future_tail.loc[i, 'sinav_donemi']

        if sinav_donemi == 1:
            mesaj = "⚠️ Sınav Dönemi Yoğunluğu: %90 üzeri doluluk riski!"
            renk = "red"
        elif tahmin_yuzdesi > 75:
            mesaj = "Yüksek Yoğunluk: Masa bulmakta zorlanabilirsiniz."
            renk = "orange"
        else:
            mesaj = "Normal Seviye: Rahatça yer bulabilirsiniz."
            renk = "green"

        formatted_data.append({
            'tarih': tarih,
            'doluluk': int(tahmin_yuzdesi),
            'guven': int(guven_araligi_yari),
            'mesaj': mesaj,
            'renk': renk
        })
    return formatted_data


PREDICTION_DATA = format_prediction_data(tahminler_df, future)

MASA_PLANLARI = [
    # Sol Üstteki 4'lü masalar
    {'id': 1, 'x': 50, 'y': 50, 'width': 120, 'height': 120, 'capacity': 4},
    {'id': 2, 'x': 200, 'y': 50, 'width': 120, 'height': 120, 'capacity': 4},
    {'id': 3, 'x': 350, 'y': 50, 'width': 120, 'height': 120, 'capacity': 4},

    # Ortadaki uzun 6'lı masalar
    {'id': 4, 'x': 50, 'y': 250, 'width': 180, 'height': 90, 'capacity': 6},
    {'id': 5, 'x': 280, 'y': 250, 'width': 180, 'height': 90, 'capacity': 6},
    {'id': 6, 'x': 510, 'y': 250, 'width': 180, 'height': 90, 'capacity': 6},

]

TOPLAM_SANDALYE = sum(m['capacity'] for m in MASA_PLANLARI)


def get_realtime_data():
    """Anlık veri simülasyonu (Her 10 dk'da bir dosyadan okuyacağınız kısım)"""

    min_kisisi = int(TOPLAM_SANDALYE * 0.4)
    max_kisisi = int(TOPLAM_SANDALYE * 0.95)
    anlik_kisi_sayisi = random.randint(min_kisisi, max_kisisi)

    doluluk_yuzdesi = round((anlik_kisi_sayisi / TOPLAM_SANDALYE) * 100, 1)

    tum_sandalyeler = [False] * TOPLAM_SANDALYE

    # Dolu sandalye sayısına göre True değerlerini listeye yerleştir
    for i in random.sample(range(TOPLAM_SANDALYE), anlik_kisi_sayisi):
        tum_sandalyeler[i] = True

    # Masa bazında boş sandalye sayısı
    bos_masa_sayisi = sum(1 for durum in tum_sandalyeler if not durum)

    # 4. Mesaj
    if doluluk_yuzdesi > 80:
        mesaj = f"⚠️ Çok Yüksek Yoğunluk! Şu an {anlik_kisi_sayisi} kişi var. Boş sandalye: {TOPLAM_SANDALYE - anlik_kisi_sayisi}"
        renk = "red"
    elif doluluk_yuzdesi > 60:
        mesaj = f"Ortalama Üzeri Yoğunluk. Boş sandalye: {TOPLAM_SANDALYE - anlik_kisi_sayisi}"
        renk = "orange"
    else:
        mesaj = f"Düşük Yoğunluk. Boş sandalye: {TOPLAM_SANDALYE - anlik_kisi_sayisi}"
        renk = "green"

    return {
        'kisi_sayisi': anlik_kisi_sayisi,
        'doluluk_yuzdesi': doluluk_yuzdesi,
        'bos_masa': TOPLAM_SANDALYE - anlik_kisi_sayisi,
        'tum_sandalyeler': tum_sandalyeler,
        'mesaj': mesaj,
        'renk': renk
    }


# --- 2. Tkinter Arayüz Sınıfı ---

class LibraryApp:
    def __init__(self, master, prediction_data):
        self.master = master
        master.title(" Kütüphane Doluluk Takip ve Tahmin Sistemi")
        # Pencereni dikey boyutu 650'den 850'ye çıkarıldı.
        master.geometry("800x850")

        self.notebook = ttk.Notebook(master)
        self.notebook.pack(pady=10, padx=10, expand=True, fill="both")

        # 1. Sekme: Anlık Durum
        self.realtime_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.realtime_frame, text=" Anlık Durum & Masa Haritası")

        # 2. Sekme: 4 Günlük Tahmin
        self.forecast_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.forecast_frame, text=" Gelecek Tahminleri")

        # Anlık Durum Sekmesini Kur
        self.setup_realtime_tab()

        # Tahmin Sekmesini Kur
        self.setup_forecast_tab(prediction_data)

        # Sandalye durumlarını tutan global bir sayaç
        self.sandalye_sayaci = 0

        # Anlık güncellemeyi başlat (10000 ms = 10 saniyede bir)
        self.master.after(100, self.update_realtime)

    # --- Anlık Durum Sekmesi (Masa Haritası) ---

    def setup_realtime_tab(self):
        summary_frame = ttk.Frame(self.realtime_frame)
        summary_frame.pack(fill='x', pady=10)

        self.kisi_label = ttk.Label(summary_frame, text="Kişi Sayısı: -", font=("Helvetica", 14, "bold"))
        self.kisi_label.pack(side='left', padx=15)

        self.doluluk_label = ttk.Label(summary_frame, text="Doluluk (%): -", font=("Helvetica", 14, "bold"))
        self.doluluk_label.pack(side='left', padx=15)

        self.bos_masa_label = ttk.Label(summary_frame, text="Boş Sandalye: -", font=("Helvetica", 14, "bold"))
        self.bos_masa_label.pack(side='left', padx=15)

        self.mesaj_label = ttk.Label(self.realtime_frame, text="Veri Yükleniyor...", font=("Helvetica", 12, "italic"))
        self.mesaj_label.pack(fill='x', pady=5)

        # 2. Masa Haritası (Canvas)
        ttk.Label(self.realtime_frame, text="Kütüphane Masa Düzeni (Yeşil: Boş / Kırmızı: Dolu)",
                  font=("Helvetica", 12)).pack(pady=5)

        self.canvas = tk.Canvas(self.realtime_frame, bg="white", height=650, width=750, highlightbackground="gray")
        self.canvas.pack(fill='both', expand=True)

        # 3. Harita İçin Etiketler
        legend_frame = ttk.Frame(self.realtime_frame)
        legend_frame.pack(pady=10)

        # Etiket Placeholder'larını güncelleyelim
        legend_canvas_green = tk.Canvas(legend_frame, width=30, height=20)
        legend_canvas_green.pack(side='left', padx=(0, 5))
        legend_canvas_green.create_oval(5, 5, 15, 15, fill='green', outline='black')
        ttk.Label(legend_frame, text="Boş Sandalye (Yeşil)").pack(side='left', padx=(0, 15))

        legend_canvas_red = tk.Canvas(legend_frame, width=30, height=20)
        legend_canvas_red.pack(side='left', padx=(0, 5))
        legend_canvas_red.create_oval(5, 5, 15, 15, fill='red', outline='black')
        ttk.Label(legend_frame, text="Dolu Sandalye (Kırmızı)").pack(side='left', padx=5)

    def update_realtime(self):
        """Anlık veriyi alır ve GUI'yi günceller (Simülasyon)."""

        realtime_data = get_realtime_data()

        self.kisi_label.config(text=f"Kişi Sayısı: {realtime_data['kisi_sayisi']}")
        self.doluluk_label.config(text=f"Doluluk (%): {realtime_data['doluluk_yuzdesi']}")
        self.bos_masa_label.config(text=f"Boş Sandalye: {TOPLAM_SANDALYE - realtime_data['kisi_sayisi']}")
        self.mesaj_label.config(text=realtime_data['mesaj'], foreground=realtime_data['renk'])

        # 2. Canvas'ı Güncelle (Masa Haritasını Çiz)
        self.draw_seating_map(realtime_data['tum_sandalyeler'])

        # Otomatik Güncelleme Planlama (Her 10 saniyede bir)
        self.master.after(10000, self.update_realtime)

    def draw_seating_map(self, tum_sandalyeler):
        """Masa haritasını Canvas'a çizer ve sandalyeleri dolu/boş işaretler."""
        self.canvas.delete("all")  # Önceki çizimleri temizle

        self.canvas.create_rectangle(20, 20, 780, 780, outline="gray", width=2)

        # Sandalye index'ini takip etmek için sayaç
        sandalye_index_current = 0

        # Masa çizimi
        for masa in MASA_PLANLARI:
            x, y = masa['x'], masa['y']
            w, h = masa['width'], masa['height']
            capacity = masa['capacity']

            # Masa (Dikdörtgen)
            masa_color = "#ccc"
            self.canvas.create_rectangle(x, y, x + w, y + h,
                                         fill=masa_color, outline="#444", width=2)

            # Masa Numarası
            self.canvas.create_text(x + w / 2, y + h / 2,
                                    text=f"Masa {masa['id']}\n({capacity} Kişilik)",
                                    fill="black", font=("Arial", 9, "bold"))

            # Sandalye Konumlarını Hesapla ve Çiz
            radius = 10

            # Sandalye koordinatları (Masayı ortalayarak)
            sandalyeler = []

            if capacity == 4:
                # 4 kişilik (2 üst, 2 alt)
                sandalyeler.append((x + w / 4, y - radius - 5))  # Üst Sol
                sandalyeler.append((x + 3 * w / 4, y - radius - 5))  # Üst Sağ
                sandalyeler.append((x + w / 4, y + h + radius + 5))  # Alt Sol
                sandalyeler.append((x + 3 * w / 4, y + h + radius + 5))  # Alt Sağ

            elif capacity == 6:
                # 6 kişilik (3 üst, 3 alt)
                sandalyeler.append((x + w / 6, y - radius - 5))  # Üst Sol
                sandalyeler.append((x + 3 * w / 6, y - radius - 5))  # Üst Orta
                sandalyeler.append((x + 5 * w / 6, y - radius - 5))  # Üst Sağ
                sandalyeler.append((x + w / 6, y + h + radius + 5))  # Alt Sol
                sandalyeler.append((x + 3 * w / 6, y + h + radius + 5))  # Alt Orta
                sandalyeler.append((x + 5 * w / 6, y + h + radius + 5))  # Alt Sağ

            # Her bir sandalyeyi çiz
            for k in range(capacity):
                if sandalye_index_current < len(tum_sandalyeler):
                    is_dolu = tum_sandalyeler[sandalye_index_current]

                    sandalye_color = "red" if is_dolu else "green"

                    cx, cy = sandalyeler[k]

                    self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                                            fill=sandalye_color, outline="black", tags="sandalye")

                    sandalye_index_current += 1

        # Danışma Masasını Çiz (Örnek görseldeki gibi)
        self.canvas.create_rectangle(650, 480, 750, 560, fill="#f0e68c", outline="#444", tags="danisma")
        self.canvas.create_text(700, 520, text="Danışma", fill="black", font=("Arial", 10, "bold"))



    def setup_forecast_tab(self, data):
        """4 Günlük Prophet tahminlerini tablo formatında gösterir."""

        ttk.Label(self.forecast_frame, text="🔮 Prophet Modelinden Gelecek 4 Günlük Doluluk Tahminleri",
                  font=("Helvetica", 16, "bold")).pack(pady=10)

        # Tahminleri göstermek için tablo
        self.tree = ttk.Treeview(self.forecast_frame,
                                 columns=('Tarih', 'Doluluk', 'Güven', 'Mesaj'),
                                 show='headings')

        self.tree.heading('Tarih', text='Tarih', anchor=tk.W)
        self.tree.heading('Doluluk', text='Tahmin (%)', anchor=tk.CENTER)
        self.tree.heading('Güven', text='Güven Aralığı', anchor=tk.CENTER)
        self.tree.heading('Mesaj', text='Tavsiye', anchor=tk.W)

        self.tree.column('Tarih', width=120)
        self.tree.column('Doluluk', width=90, anchor=tk.CENTER)
        self.tree.column('Güven', width=120, anchor=tk.CENTER)
        self.tree.column('Mesaj', width=350)

        self.tree.pack(fill='both', expand=True, pady=10, padx=10)

        self.load_forecast_data(data)

    def load_forecast_data(self, data):
        """Prophet tahminlerini tabloya yükler."""

        # Renk etiketleri
        self.tree.tag_configure('red', foreground='red')
        self.tree.tag_configure('orange', foreground='orange')
        self.tree.tag_configure('green', foreground='green')

        for item in data:
            guven_str = f"±%{item['guven']}"
            self.tree.insert('', tk.END,
                             values=(item['tarih'], item['doluluk'], guven_str, item['mesaj']),
                             tags=(item['renk'],))


# --- Uygulamayı Çalıştırma ---
if __name__ == "__main__":
    root = tk.Tk()
    app = LibraryApp(root, PREDICTION_DATA)
    root.mainloop()
