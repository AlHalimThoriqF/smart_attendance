from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.database.database import get_db
from app import schemas
from app import repositories
from app.core.security import get_current_user
from app.ai.background_monitor import BackgroundMonitor

router = APIRouter(prefix="/api/cctv", tags=["cctv"])

@router.get("", response_model=List[schemas.CCTVResponse])
def get_all_cctv(db: Session = Depends(get_db), admin=Depends(get_current_user)):
    return repositories.cctv.get_all_cctvs(db)

@router.get("/{camera_id}", response_model=schemas.CCTVResponse)
def get_cctv(camera_id: int, db: Session = Depends(get_db), admin=Depends(get_current_user)):
    db_camera = repositories.cctv.get_cctv(db, camera_id)
    if not db_camera:
        raise HTTPException(status_code=404, detail="CCTV camera not found")
    return db_camera

@router.post("", response_model=schemas.CCTVResponse, status_code=status.HTTP_201_CREATED)
def create_cctv(camera: schemas.CCTVCreate, db: Session = Depends(get_db), admin=Depends(get_current_user)):
    db_cctv = repositories.cctv.create_cctv(db, camera)
    if db_cctv.status:
        rtsp = db_cctv.rtsp_url
        if rtsp.isdigit():
            rtsp = int(rtsp)
        BackgroundMonitor.start_camera(db_cctv.id, rtsp)
    return db_cctv

@router.put("/{camera_id}", response_model=schemas.CCTVResponse)
def update_cctv(camera_id: int, camera_data: schemas.CCTVUpdate, db: Session = Depends(get_db), admin=Depends(get_current_user)):
    db_camera = repositories.cctv.get_cctv(db, camera_id)
    if not db_camera:
        raise HTTPException(status_code=404, detail="CCTV camera not found")
    updated_cctv = repositories.cctv.update_cctv(db, db_camera, camera_data)
    
    # Stop the existing camera stream to force a restart with updated configurations
    BackgroundMonitor.stop_camera(updated_cctv.id)
    
    if updated_cctv.status:
        rtsp = updated_cctv.rtsp_url
        if rtsp.isdigit():
            rtsp = int(rtsp)
        BackgroundMonitor.start_camera(updated_cctv.id, rtsp)
        
    return updated_cctv

@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_cctv(camera_id: int, db: Session = Depends(get_db), admin=Depends(get_current_user)):
    db_camera = repositories.cctv.get_cctv(db, camera_id)
    if not db_camera:
        raise HTTPException(status_code=404, detail="CCTV camera not found")
    repositories.cctv.delete_cctv(db, db_camera)
    BackgroundMonitor.stop_camera(camera_id)
    return None
