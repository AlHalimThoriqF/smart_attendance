from datetime import datetime
from pydantic import BaseModel
from typing import Optional

from .lecture import LectureResponse

class DetectionLogBase(BaseModel):
    cctv_id: int
    lecture_id: int
    confidence: float
    status: str = "present"

class DetectionLogResponse(BaseModel):
    id: int
    cctv_id: int
    lecture_id: int
    confidence: float
    first_seen: datetime
    last_seen: datetime
    status: str
    snapshot_path: Optional[str] = None
    last_snapshot_path: Optional[str] = None
    crop_snapshot_path: Optional[str] = None
    lecture: LectureResponse

    class Config:
        orm_mode = True
