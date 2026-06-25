
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.database.database import engine, Base
from app.api import logs, stream, upload
from app import pages
from app.ai.recognition import initialize_models
from app.ai.background_monitor import BackgroundMonitor
import os
import time
import logging
from sqlalchemy.exc import OperationalError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Attendance",
    description="Civitas Attendance Monitoring",
)

@app.on_event("startup")
def on_startup():
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
    # Initialize AI models (InsightFace & SVM)
    initialize_models()
    BackgroundMonitor.start_all()

# Mount static files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Mount faces storage
faces_dir = os.path.join(BASE_DIR, "storage", "faces")
os.makedirs(faces_dir, exist_ok=True)
app.mount("/faces", StaticFiles(directory=faces_dir), name="faces")

# Mount snapshots storage
snapshots_dir = os.path.join(BASE_DIR, "storage", "snapshots")
os.makedirs(snapshots_dir, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=snapshots_dir), name="snapshots")

# Mount videos storage
videos_dir = os.path.join(BASE_DIR, "storage", "videos")
os.makedirs(videos_dir, exist_ok=True)
app.mount("/videos", StaticFiles(directory=videos_dir), name="videos")

# Include frontend pages router
app.include_router(pages.router)
# Include API and WebSocket routers

app.include_router(logs.router)
app.include_router(stream.router)
app.include_router(upload.router)
