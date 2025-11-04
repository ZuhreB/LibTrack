import mysql.connector

AYARLAR = {
    "host": "localhost",
    "user": "root",
    "password": "zuhre060",
    "database": "newschema"  # yeni db açarız burası değişir
}

try:
    conn = mysql.connector.connect(**AYARLAR)

    if conn.is_connected():
        print("MySQL veritabanına başarıyla bağlanıldı")

        cursor = conn.cursor()
        cursor.execute("SELECT VERSION()")
        db_versiyon = cursor.fetchone()
        print(f"MySQL Veritabanı Versiyonu: {db_versiyon[0]}")

except mysql.connector.Error as err:
    print(f"Bağlantı Hatası: {err}")


finally:

    if 'conn' in locals() and conn.is_connected():
        cursor.close()
        conn.close()
        print("MySQL bağlantısı kapatıldı.")