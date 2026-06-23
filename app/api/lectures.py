import os
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
from app.database.database import get_db
from app import schemas
from app import repositories
from app import schemas, models, repositories

router = APIRouter(prefix="/api/civitas", tags=["civitas"])
logs_router = APIRouter(prefix="/api/detection-logs", tags=["detection-logs"])

@router.get("", response_model=List[schemas.LectureResponse])
def get_all_lectures(db: Session = Depends(get_db)):
    return repositories.lectures.get_all_lectures(db)

@router.post("", response_model=schemas.LectureResponse, status_code=status.HTTP_201_CREATED)
async def register_lecture(
    identifier: str = Form(...),
    name: str = Form(...),
    gender: str = Form("Unknown"),
    jabatan: Optional[str] = Form(None),
    program_studi: Optional[str] = Form(None),
    jabatan_struktural: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    existing_lecture = repositories.lectures.get_lecture_by_nis(db, identifier)
    if existing_lecture:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Civitas member with identifier '{identifier}' is already registered."
        )

    import re
    safe_name = re.sub(r'[\\/*?:"<>|]', "", name).strip()
    
    filename = None
    if file:
        file_extension = os.path.splitext(file.filename)[1] or ".jpg"
        filename = f"{safe_name}{file_extension}"

    db_lecture = repositories.lectures.create_lecture(db, identifier, name, gender, jabatan, program_studi, jabatan_struktural, filename)

    if file:
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

@router.put("/{lecture_id}", response_model=schemas.LectureResponse)
async def update_lecture(
    lecture_id: int,
    identifier: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    jabatan: Optional[str] = Form(None),
    program_studi: Optional[str] = Form(None),
    jabatan_struktural: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    db_lecture = db.query(repositories.lectures.models.Lecture).filter(repositories.lectures.models.Lecture.id == lecture_id).first()
    if not db_lecture:
        raise HTTPException(status_code=404, detail="Civitas member not found")

    if identifier and identifier != db_lecture.nis:
        existing = repositories.lectures.get_lecture_by_nis(db, identifier)
        if existing:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Identifier already in use")

    import re
    new_name = name if name else db_lecture.name
    safe_name = re.sub(r'[\\/*?:"<>|]', "", new_name).strip()
    
    filename = db_lecture.images

    if file:
        file_extension = os.path.splitext(file.filename)[1] or ".jpg"
        filename = f"{safe_name}{file_extension}"
        file_path = os.path.join(repositories.lectures.FACES_DIR, filename)
        
        # Remove old image if filename changed
        if db_lecture.images:
            old_path = os.path.join(repositories.lectures.FACES_DIR, db_lecture.images)
            if old_path != file_path and os.path.exists(old_path):
                try:
                    os.remove(old_path)
                except:
                    pass

        try:
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to write image: {str(e)}")
    else:
        # Jika tidak upload foto baru, pastikan nama file fisik sesuai dengan format Nama
        if db_lecture.images:
            old_ext = os.path.splitext(db_lecture.images)[1] or ".jpg"
            new_filename = f"{safe_name}{old_ext}"
            
            if db_lecture.images != new_filename:
                old_path = os.path.join(repositories.lectures.FACES_DIR, db_lecture.images)
                new_path = os.path.join(repositories.lectures.FACES_DIR, new_filename)
                
                if os.path.exists(old_path):
                    try:
                        os.rename(old_path, new_path)
                        filename = new_filename
                    except Exception as e:
                        print(f"Error renaming file: {e}")
                else:
                    # File fisik tidak ada, kita update field db saja (atau biarkan)
                    filename = new_filename

    updated_lecture = repositories.lectures.update_lecture(
        db=db,
        db_lecture=db_lecture,
        nis=identifier,
        name=name,
        gender=gender,
        jabatan=jabatan,
        program_studi=program_studi,
        jabatan_struktural=jabatan_struktural,
        images=filename
    )
    return updated_lecture

@router.delete("/{lecture_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_lecture(lecture_id: int, db: Session = Depends(get_db)):
    db_lecture = db.query(repositories.lectures.models.Lecture).filter(repositories.lectures.models.Lecture.id == lecture_id).first()
    if not db_lecture:
        raise HTTPException(status_code=404, detail="Civitas member not found")

    repositories.lectures.delete_lecture(db, db_lecture)
    return None

from sqlalchemy import cast, Date
from datetime import datetime

@router.get("/attendance", response_model=List[schemas.AttendanceSummary])
def get_attendance(date: str, db: Session = Depends(get_db)):
    all_lectures = repositories.lectures.get_all_lectures(db)
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
        
    from app.models.detection_log import DetectionLog
    from app.config.cctv_config import get_cctv_by_id
    
    start_of_day = datetime.combine(target_date, datetime.min.time())
    end_of_day = datetime.combine(target_date, datetime.max.time())
    
    logs_for_date = db.query(DetectionLog).filter(
        DetectionLog.first_seen >= start_of_day,
        DetectionLog.first_seen <= end_of_day
    ).all()
    
    attendance_map = {}
    for log in logs_for_date:
        if log.lecture_id is None:
            continue
        lid = log.lecture_id
        
        cctv_info = get_cctv_by_id(log.cctv_id)
        cctv_name = cctv_info['name'] if cctv_info else f"CCTV #{log.cctv_id}"
        
        if lid not in attendance_map:
            attendance_map[lid] = {
                "first_seen": log.first_seen,
                "last_seen": log.last_seen,
                "snapshot_path": log.snapshot_path,
                "cctv_name": cctv_name
            }
        else:
            if log.last_seen > attendance_map[lid]["last_seen"]:
                attendance_map[lid]["last_seen"] = log.last_seen
                attendance_map[lid]["cctv_name"] = cctv_name
            if log.first_seen < attendance_map[lid]["first_seen"]:
                attendance_map[lid]["first_seen"] = log.first_seen
                attendance_map[lid]["snapshot_path"] = log.snapshot_path

    summaries = []
    for lec in all_lectures:
        if lec.id in attendance_map:
            data = attendance_map[lec.id]
            summaries.append(
                schemas.AttendanceSummary(
                    lecture=schemas.LectureResponse(
                        id=lec.id,
                        nis=lec.nis,
                        name=lec.name,
                        gender=lec.gender,
                        jabatan=lec.jabatan,
                        program_studi=lec.program_studi,
                        jabatan_struktural=lec.jabatan_struktural,
                        images=lec.images,
                        created_at=lec.created_at
                    ),
                    status="Hadir",
                    first_seen=data["first_seen"],
                    last_seen=data["last_seen"],
                    snapshot_path=data["snapshot_path"],
                    cctv_name=data["cctv_name"]
                )
            )
        else:
            summaries.append(
                schemas.AttendanceSummary(
                    lecture=schemas.LectureResponse(
                        id=lec.id,
                        nis=lec.nis,
                        name=lec.name,
                        gender=lec.gender,
                        jabatan=lec.jabatan,
                        program_studi=lec.program_studi,
                        jabatan_struktural=lec.jabatan_struktural,
                        images=lec.images,
                        created_at=lec.created_at
                    ),
                    status="Tidak Hadir",
                    first_seen=None,
                    last_seen=None,
                    snapshot_path=None,
                    cctv_name=None
                )
            )
            
    return summaries

@logs_router.get("", response_model=List[schemas.DetectionLogResponse])
def get_all_logs(db: Session = Depends(get_db)):
    return repositories.lectures.get_all_logs(db)
