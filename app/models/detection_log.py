import datetime
from sqlalchemy import DateTime, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from app.database.database import Base
class DetectionLog(Base):
    __tablename__ = "detection_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    cctv_id: Mapped[int] = mapped_column(nullable=True)
    cctv_name: Mapped[str] = mapped_column(String(100), nullable=True)
    person_name: Mapped[str] = mapped_column(String(255), nullable=False)
    first_seen : Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(),nullable=False)
    last_seen : Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(),nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="present", nullable=False)
    snapshot_path: Mapped[str] = mapped_column(String(255), nullable=True)
    crop_snapshot_path: Mapped[str] = mapped_column(String(255), nullable=True)

    @property
    def display_cctv_name(self) -> str:
        if self.cctv_name:
            return self.cctv_name
        if self.cctv_id is not None:
            from app.config.cctv_config import get_cctv_by_id
            cctv = get_cctv_by_id(self.cctv_id)
            return cctv['name'] if cctv else f"CCTV #{self.cctv_id}"
        return "Unknown CCTV"
