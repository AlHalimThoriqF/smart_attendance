import os
import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func
from app import models

FACES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "faces")
os.makedirs(FACES_DIR, exist_ok=True)

def get_all_lectures(db: Session):
    return db.query(models.Lecture).all()

def get_lecture_by_nis(db: Session, nis: str):
    return db.query(models.Lecture).filter(models.Lecture.nis == nis).first()

def get_lecture_by_name(db: Session, name: str):
    return db.query(models.Lecture).filter(models.Lecture.name == name).first()

def create_lecture(db: Session, nis: str, name: str, gender: str = "Unknown", jabatan: str = None, program_studi: str = None, jabatan_struktural: str = None, images: str = None):
    db_lecture = models.Lecture(nis=nis, name=name, gender=gender, jabatan=jabatan, program_studi=program_studi, jabatan_struktural=jabatan_struktural, images=images)
    db.add(db_lecture)
    db.commit()
    db.refresh(db_lecture)
    return db_lecture

def update_lecture(db: Session, db_lecture: models.Lecture, nis: str = None, name: str = None, gender: str = None, jabatan: str = None, program_studi: str = None, jabatan_struktural: str = None, images: str = None):
    if nis is not None:
        db_lecture.nis = nis
    if name is not None:
        db_lecture.name = name
    if gender is not None:
        db_lecture.gender = gender
    if jabatan is not None:
        db_lecture.jabatan = jabatan
    if program_studi is not None:
        db_lecture.program_studi = program_studi
    if jabatan_struktural is not None:
        db_lecture.jabatan_struktural = jabatan_struktural
    if images is not None:
        db_lecture.images = images
        
    db.commit()
    db.refresh(db_lecture)
    return db_lecture

def delete_lecture(db: Session, db_lecture: models.Lecture):
    # Clean up associated face photo from local storage
    import re
    safe_name = re.sub(r'[\\/*?:"<>|]', "", db_lecture.name).strip()
    for ext in [".jpg", ".png", ".jpeg"]:
        file_path = os.path.join(FACES_DIR, f"{safe_name}{ext}")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")

    db.delete(db_lecture)
    db.commit()

def get_lecture_by_id(db: Session, lecture_id: int):
    return db.query(models.Lecture).filter(models.Lecture.id == lecture_id).first()

def create_detection_log(db: Session, cctv_id: int, lecture_id: int, confidence: float, status: str = "present", snapshot_path: str = None, last_snapshot_path: str = None):
    # Jika orang yang sama terekam di kamera yang sama dan belum lewat dari 30 menit sejak terakhir terlihat,
    # maka gabungkan ke dalam log yang sama (sesi yang sama).
    session_timeout = datetime.datetime.now() - datetime.timedelta(minutes=30)
    
    existing_log = db.query(models.DetectionLog).filter(
        models.DetectionLog.cctv_id == cctv_id,
        models.DetectionLog.lecture_id == lecture_id,
        models.DetectionLog.last_seen >= session_timeout
    ).order_by(models.DetectionLog.last_seen.desc()).first()

    if existing_log:
        existing_log.last_seen = func.now()
        # Perbarui nilai confidence agar selalu mencerminkan deteksi terakhir (Last Seen)
        existing_log.confidence = confidence
        
        # Update last_snapshot_path jika diberikan
        if last_snapshot_path:
            existing_log.last_snapshot_path = last_snapshot_path
            
        db.commit()
        db.refresh(existing_log)
        return existing_log
    else:
        db_log = models.DetectionLog(
            cctv_id=cctv_id,
            lecture_id=lecture_id,
            confidence=confidence,
            status=status,
            snapshot_path=snapshot_path,
            last_snapshot_path=last_snapshot_path
        )
        db.add(db_log)
        db.commit()
        db.refresh(db_log)
        return db_log

def get_all_logs(db: Session):
    return db.query(models.DetectionLog).order_by(models.DetectionLog.first_seen.desc()).all()
