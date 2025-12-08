import pandas as pd
from tkinter import messagebox
from datetime import datetime
try:
    import mysql.connector

    HAS_MYSQL_CONNECTOR = True
except ImportError:
    HAS_MYSQL_CONNECTOR = False


class LibraryDataManager:
    def __init__(self, csv_path, db_config):
        self.csv_path = csv_path
        self.db_config = db_config
        self.df = None
        self.hourly_data = None
        self.min_date = None
        self.max_date = None
        self.load_csv_data()

    def load_csv_data(self):

        try:
            self.df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            messagebox.showerror("Hata", f"{self.csv_path} dosyası bulunamadı.")
            exit()

        # Tarih formatlama ve temizleme
        self.df["datetime"] = pd.to_datetime(self.df["date"] + " " + self.df["time"])

        self.hourly_data = self.df[self.df["saatlik_ortalama_doluluk"].notnull()].copy()
        self.hourly_data["saatlik_ortalama_doluluk"] = self.hourly_data["saatlik_ortalama_doluluk"].astype(float)
        self.hourly_data["date"] = pd.to_datetime(self.hourly_data["date"])
        self.hourly_data["hour"] = self.hourly_data["datetime"].dt.hour
        self.hourly_data["weekday"] = self.hourly_data["datetime"].dt.weekday

        self.min_date = self.hourly_data["date"].min().date()
        self.max_date = self.hourly_data["date"].max().date()

    def fetch_live_occupancy(self):

        if not HAS_MYSQL_CONNECTOR:
            return "Bağlantı Hatası (Kütüphane Yok)"

        try:
            connection = mysql.connector.connect(**self.db_config)
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
                return float(result[0])
            else:
                return "Veri Yok"

        except mysql.connector.Error as err:
            print(f"Veritabanı Hatası: {err}")
            return f"DB Hata: {err.errno}"
        except Exception as e:
            return f"Hata: {str(e)[:15]}..."