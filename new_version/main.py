from new_version.gui_app import LibTrackApp
from data_manager import LibraryDataManager
from forecasting_engine import ForecastingEngine

if __name__ == "__main__":
    CSV_FILE_PATH = "../libtrack_dataset_bounded_realistic_v2.csv"
    LIBRARY_CAPACITY = 432
    DATABASE_CONFIG = {
        "host": "localhost",
        "user": "root",
        "password": "zuhre060",
        "database": "yolo_db"
    }

    print("Uygulama başlatılıyor...")

    # 1. Veriyi Yükle
    data_mgr = LibraryDataManager(CSV_FILE_PATH, DATABASE_CONFIG)

    # 2. Modelleri Hazırla
    forecaster = ForecastingEngine(LIBRARY_CAPACITY)

    # 3. Uygulamayı Başlat
    app = LibTrackApp(data_manager=data_mgr, forecasting_engine=forecaster)

    app.mainloop()