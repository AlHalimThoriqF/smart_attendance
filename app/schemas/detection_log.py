from datetime import datetime
from pydantic import BaseModel
from typing import Optional

from .lecture import LectureResponse
from .cctv import CCTVResponse

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
    lecture: LectureResponse
    cctv: Optional[CCTVResponse] = None

    class Config:
        orm_mode = True
