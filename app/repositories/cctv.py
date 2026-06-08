from sqlalchemy.orm import Session
from app import models, schemas

def get_all_cctvs(db: Session):
    return db.query(models.CCTV).all()

def get_cctv(db: Session, camera_id: int):
    return db.query(models.CCTV).filter(models.CCTV.id == camera_id).first()

def create_cctv(db: Session, camera: schemas.CCTVCreate):
    db_camera = models.CCTV(**camera.dict())
    db.add(db_camera)
    db.commit()
    db.refresh(db_camera)
    return db_camera

def update_cctv(db: Session, db_camera: models.CCTV, camera_data: schemas.CCTVUpdate):
    for key, value in camera_data.dict(exclude_unset=True).items():
        setattr(db_camera, key, value)
    db.commit()
    db.refresh(db_camera)
    return db_camera

def delete_cctv(db: Session, db_camera: models.CCTV):
    db.delete(db_camera)
    db.commit()
