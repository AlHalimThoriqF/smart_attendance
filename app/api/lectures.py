import os
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List
from app.database.database import get_db
from app import schemas
from app import repositories
from app.core.security import get_current_user

router = APIRouter(prefix="/api/users", tags=["users"])
logs_router = APIRouter(prefix="/api/detection-logs", tags=["detection-logs"])

@router.get("", response_model=List[schemas.LectureResponse])
def get_all_lectures(db: Session = Depends(get_db)):
    return repositories.lectures.get_all_lectures(db)

@router.post("", response_model=schemas.LectureResponse, status_code=status.HTTP_201_CREATED)
async def register_lecture(
    identifier: str = Form(...),
    name: str = Form(...),
    gender: str = Form("Unknown"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    admin=Depends(get_current_user)
):
    existing_lecture = repositories.lectures.get_lecture_by_nis(db, identifier)
    if existing_lecture:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Civitas member with identifier '{identifier}' is already registered."
        )

    file_extension = os.path.splitext(file.filename)[1] or ".jpg"
    filename = f"{identifier}{file_extension}"

    db_lecture = repositories.lectures.create_lecture(db, identifier, name, gender, filename)

    file_path = os.path.join(repositories.lectures.FACES_DIR, filename)
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        repositories.lectures.delete_lecture(db, db_lecture)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to write face registration photo to disk: {str(e)}"
        )

    return db_lecture

@router.delete("/{lecture_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_lecture(lecture_id: int, db: Session = Depends(get_db), admin=Depends(get_current_user)):
    db_lecture = db.query(repositories.lectures.models.Lecture).filter(repositories.lectures.models.Lecture.id == lecture_id).first()
    if not db_lecture:
        raise HTTPException(status_code=404, detail="Civitas member not found")

    repositories.lectures.delete_lecture(db, db_lecture)
    return None



@logs_router.get("", response_model=List[schemas.DetectionLogResponse])
def get_all_logs(db: Session = Depends(get_db), admin=Depends(get_current_user)):
    return repositories.lectures.get_all_logs(db)
