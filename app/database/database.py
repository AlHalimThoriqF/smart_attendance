import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Load environment variables from .env
load_dotenv()

# Gunakan SQLite secara paksa untuk aplikasi portabel
DATABASE_URL = "sqlite:///./smart_attendance.db"

# Create SQLAlchemy engine
# connect_args={"check_same_thread": False} diperlukan oleh SQLite jika digunakan di multi-threading (seperti AI Background Monitor)
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# Event Listener untuk mengaktifkan mode WAL (Write-Ahead Logging) di SQLite
# WAL mode mencegah error "database is locked" saat beberapa thread mencoba menulis ke SQLite secara bersamaan
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()

# Configure the sessionmaker
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Define standard DeclarativeBase for SQLAlchemy 2.0+
class Base(DeclarativeBase):
    pass

# FastAPI Dependency to get database sessions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
