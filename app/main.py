
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.database.database import engine, Base, SessionLocal
from app.api import cctv, lectures, stream, auth
from app import pages
from app.repositories.users import seed_default_user
from app.ai.recognition import initialize_models
from app.ai.background_monitor import BackgroundMonitor
import os
import time
import logging
from sqlalchemy.exc import OperationalError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    title="Smart Attendance",
    description="FastAPI for Real-time CCTV Stream processing and Civitas Attendance Monitoring",
)

@app.on_event("startup")
def on_startup():
    # Wait for database connection
    while True:
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database connected and tables verified.")
            break
        except OperationalError as e:
            logger.warning("Database connection failed. Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            logger.warning(f"Unexpected error: {e}. Retrying in 5 seconds...")
            time.sleep(5)

    db = SessionLocal()
    try:
        seed_default_user(db)
    finally:
        db.close()
        
    # Initialize AI models (InsightFace & SVM)
    initialize_models()
    
    # Start Background Monitoring for all active CCTV cameras
    BackgroundMonitor.start_all()

# Mount static files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include frontend pages router
app.include_router(pages.router)

# Include API and WebSocket routers
app.include_router(auth.router)
app.include_router(cctv.router)
app.include_router(lectures.router)
app.include_router(lectures.logs_router)
app.include_router(stream.router)
