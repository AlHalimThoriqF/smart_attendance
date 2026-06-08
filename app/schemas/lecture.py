from datetime import datetime
from pydantic import BaseModel
from typing import Optional

class LectureBase(BaseModel):
    nis: str
    name: str
    gender: str
    jabatan: Optional[str] = None
    program_studi: Optional[str] = None
    jabatan_struktural: Optional[str] = None
    images: Optional[str] = None

class LectureCreate(LectureBase):
    pass

class LectureResponse(LectureBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True
