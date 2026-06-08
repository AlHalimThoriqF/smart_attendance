import os
from sqlalchemy.orm import Session
from app import models

FACES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "faces")
os.makedirs(FACES_DIR, exist_ok=True)

def get_all_lectures(db: Session):
    return db.query(models.Lecture).all()

def get_lecture_by_nis(db: Session, nis: str):
    return db.query(models.Lecture).filter(models.Lecture.nis == nis).first()

def create_lecture(db: Session, nis: str, name: str, gender: str = "Unknown", images: str = None):
    db_lecture = models.Lecture(nis=nis, name=name, gender=gender, images=images)
    db.add(db_lecture)
    db.commit()
    db.refresh(db_lecture)
    return db_lecture

def delete_lecture(db: Session, db_lecture: models.Lecture):
    # Clean up associated face photo from local storage
    for ext in [".jpg", ".png", ".jpeg"]:
        file_path = os.path.join(FACES_DIR, f"{db_lecture.nis}{ext}")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")

    db.delete(db_lecture)
    db.commit()

def get_lecture_by_id(db: Session, lecture_id: int):
    return db.query(models.Lecture).filter(models.Lecture.id == lecture_id).first()

def create_detection_log(db: Session, cctv_id: int, lecture_id: int, confidence: float, status: str = "present"):
    db_log = models.DetectionLog(
        cctv_id=cctv_id,
        lecture_id=lecture_id,
        confidence=confidence,
        status=status
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log

def get_all_logs(db: Session):
    return db.query(models.DetectionLog).order_by(models.DetectionLog.first_seen.desc()).all()
