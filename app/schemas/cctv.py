from datetime import datetime
from pydantic import BaseModel
from typing import Optional

class CCTVBase(BaseModel):
    name: str
    location: str
    rtsp_url: str
    status: bool = True

class CCTVCreate(CCTVBase):
    pass

class CCTVUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    rtsp_url: Optional[str] = None
    status: Optional[bool] = None

class CCTVResponse(CCTVBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True
