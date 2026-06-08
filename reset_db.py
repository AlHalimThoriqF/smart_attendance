from app.database.database import engine, Base
from sqlalchemy import text
import app.models

print("Dropping all existing tables...")
Base.metadata.drop_all(bind=engine)
with engine.connect() as conn:
    conn.execute(text('DROP TABLE IF EXISTS alembic_version'))
    conn.commit()

print("Database cleared!")
