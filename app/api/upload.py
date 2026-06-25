from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks
import os
import datetime
import shutil
import uuid
from app.ai.upload_processor import VideoUploadProcessor
from app.config.cctv_config import CCTVS

router = APIRouter(prefix="/api/upload", tags=["upload"])

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "videos")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/video")
def upload_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    start_time: str = Form(...),
    cctv_name: str = Form(...)
):
    try:
        # Menangani format 'YYYY-MM-DDTHH:MM'
        parsed_time = datetime.datetime.fromisoformat(start_time)
    except ValueError:
        return {"success": False, "message": "Format waktu tidak valid. Gunakan format ISO."}

    file_extension = os.path.splitext(video.filename)[1]
    if not file_extension:
        file_extension = ".mp4"
        
    safe_filename = f"upload_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    cctv_id = None
    for c in CCTVS:
        if c['name'].lower() == cctv_name.lower():
            cctv_id = c['id']
            break
            
    if cctv_id is None:
        cctv_id = 0

    # Memproses video di background agar response instan
    task_id = str(uuid.uuid4())
    background_tasks.add_task(VideoUploadProcessor.process_uploaded_video, file_path, parsed_time, cctv_name, cctv_id, task_id)

    return {"success": True, "message": "Video berhasil diunggah dan sedang diproses.", "task_id": task_id}

@router.get("/status/{task_id}")
async def get_upload_status(task_id: str):
    status = VideoUploadProcessor.upload_progress.get(task_id)
    if not status:
        return {"status": "not_found", "message": "Task tidak ditemukan"}
    return status

@router.post("/cancel/{task_id}")
async def cancel_upload(task_id: str):
    if task_id in VideoUploadProcessor.upload_progress:
        VideoUploadProcessor.upload_progress[task_id]['status'] = 'cancelled'
        VideoUploadProcessor.upload_progress[task_id]['message'] = 'Dibatalkan oleh pengguna.'
        return {"success": True, "message": "Proses dibatalkan."}
    return {"success": False, "message": "Task tidak ditemukan atau sudah selesai."}
