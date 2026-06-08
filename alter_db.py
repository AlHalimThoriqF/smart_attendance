import os
from dotenv import load_dotenv
import pymysql

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_DATABASE = os.getenv("DB_DATABASE", "smart_attendance")
DB_USERNAME = os.getenv("DB_USERNAME", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

try:
    connection = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USERNAME,
        password=DB_PASSWORD,
        database=DB_DATABASE
    )
    with connection.cursor() as cursor:
        try:
            cursor.execute("ALTER TABLE lectures ADD COLUMN jabatan VARCHAR(100) NULL;")
            print("Added jabatan column.")
        except Exception as e:
            print("Error adding jabatan (might already exist):", e)
            
        try:
            cursor.execute("ALTER TABLE lectures ADD COLUMN program_studi VARCHAR(150) NULL;")
            print("Added program_studi column.")
        except Exception as e:
            print("Error adding program_studi (might already exist):", e)
            
        try:
            cursor.execute("ALTER TABLE lectures ADD COLUMN jabatan_struktural VARCHAR(150) NULL;")
            print("Added jabatan_struktural column.")
        except Exception as e:
            print("Error adding jabatan_struktural (might already exist):", e)
            
    connection.commit()
    print("Database altered successfully.")
except Exception as e:
    print(f"Failed to connect or alter database: {e}")
