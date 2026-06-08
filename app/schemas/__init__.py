from .cctv import CCTVBase, CCTVCreate, CCTVUpdate, CCTVResponse
from .user import UserBase, UserCreate, UserResponse
from .lecture import LectureBase, LectureCreate, LectureResponse
from .detection_log import DetectionLogBase, DetectionLogResponse
from .auth import Token, TokenData

__all__ = [
    "CCTVBase", "CCTVCreate", "CCTVUpdate", "CCTVResponse",
    "UserBase", "UserCreate", "UserResponse",
    "LectureBase", "LectureCreate", "LectureResponse",
    "DetectionLogBase", "DetectionLogResponse",
    "Token", "TokenData"
]
