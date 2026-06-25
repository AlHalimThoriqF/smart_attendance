from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from app.database.database import get_db
from app import repositories
from app.schemas.detection_log import DetectionLogResponse

router = APIRouter(prefix="/api/detection-logs", tags=["logs"])

@router.get("", response_model=List[DetectionLogResponse])
def get_all_logs(db: Session = Depends(get_db)):
    return repositories.logs.get_all_logs(db)

@router.get("/person/{person_name}", response_model=List[DetectionLogResponse])
def get_logs_by_person(person_name: str, db: Session = Depends(get_db)):
    return repositories.logs.get_logs_by_person_name(db, person_name)
