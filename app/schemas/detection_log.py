from datetime import datetime
from pydantic import BaseModel
from typing import Optional

class DetectionLogBase(BaseModel):
    cctv_id: int
    person_name: str
    confidence: float
    status: str = "present"

class DetectionLogResponse(BaseModel):
    id: int
    cctv_id: Optional[int] = None
    person_name: str
    confidence: float
    first_seen: datetime
    last_seen: datetime
    status: str
    snapshot_path: Optional[str] = None
    crop_snapshot_path: Optional[str] = None
    cctv_name: Optional[str] = None
    display_cctv_name: Optional[str] = None

    class Config:
        from_attributes = True
