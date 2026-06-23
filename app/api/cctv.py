from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from app.config.cctv_config import get_all_cctvs, get_cctv_by_id
from app.ai.background_monitor import BackgroundMonitor

router = APIRouter(prefix="/api/cctv", tags=["cctv"])

@router.get("")
def get_all_cctv():
    return get_all_cctvs()

@router.get("/{camera_id}")
def get_cctv(camera_id: int):
    camera = get_cctv_by_id(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="CCTV camera not found")
    return camera

@router.post("/{camera_id}/refresh")
def refresh_cctv(camera_id: int):
    camera = get_cctv_by_id(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="CCTV camera not found")
        
    BackgroundMonitor.stop_camera(camera['id'])
    if camera['status']:
        rtsp = camera['rtsp_url']
        if str(rtsp).isdigit():
            rtsp = int(rtsp)
        BackgroundMonitor.start_camera(camera['id'], rtsp)
        
    return camera
