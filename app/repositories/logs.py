import os
import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func
from app import models

def create_detection_log(db: Session, cctv_id: int, person_name: str, confidence: float, snapshot_path: str = None, crop_snapshot_path: str = None, status: str = "present", cctv_name: str = None, timestamp_override: datetime.datetime = None):
    # Jika orang yang sama terekam di kamera yang sama dan belum lewat dari 30 menit sejak terakhir terlihat,
    # maka gabungkan ke dalam log yang sama (sesi yang sama).
    
    current_time = timestamp_override if timestamp_override else datetime.datetime.now()
    session_timeout = current_time - datetime.timedelta(minutes=5)
    
    existing_log = db.query(models.DetectionLog).filter(
        models.DetectionLog.cctv_id == cctv_id,
        models.DetectionLog.person_name == person_name,
        models.DetectionLog.last_seen >= session_timeout,
        models.DetectionLog.last_seen <= current_time
    ).order_by(models.DetectionLog.last_seen.desc()).first()

    if existing_log:
        existing_log.last_seen = current_time
        # Perbarui nilai confidence dan snapshot jika confidence yang baru lebih tinggi
        if confidence > existing_log.confidence:
            existing_log.confidence = confidence
            
            if snapshot_path:
                existing_log.snapshot_path = snapshot_path

            if crop_snapshot_path:
                existing_log.crop_snapshot_path = crop_snapshot_path
                
        if cctv_name:
            existing_log.cctv_name = cctv_name
            
        db.commit()
        db.refresh(existing_log)
        return existing_log
    else:
        db_log = models.DetectionLog(
            cctv_id=cctv_id,
            cctv_name=cctv_name,
            person_name=person_name,
            confidence=confidence,
            status=status,
            snapshot_path=snapshot_path,

            crop_snapshot_path=crop_snapshot_path,
            first_seen=current_time,
            last_seen=current_time
        )
        db.add(db_log)
        db.commit()
        db.refresh(db_log)
        return db_log

def get_all_logs(db: Session):
    return db.query(models.DetectionLog).order_by(models.DetectionLog.first_seen.desc()).all()

def get_log_by_id(db: Session, log_id: int):
    return db.query(models.DetectionLog).filter(models.DetectionLog.id == log_id).first()

def get_logs_by_person_name(db: Session, person_name: str):
    return db.query(models.DetectionLog).filter(models.DetectionLog.person_name == person_name).order_by(models.DetectionLog.first_seen.desc()).all()
