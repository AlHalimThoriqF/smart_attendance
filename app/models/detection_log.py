import datetime
from sqlalchemy import DateTime, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from app.database.database import Base
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.cctv import CCTV
    from app.models.lecture import Lecture

class DetectionLog(Base):
    __tablename__ = "detection_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    cctv_id: Mapped[int] = mapped_column(
        ForeignKey("cctv.id", ondelete="CASCADE"),
        nullable=False
    )
    lecture_id: Mapped[int] = mapped_column(
        ForeignKey("lectures.id", ondelete="CASCADE"),
        nullable=False  # Strict rule: Every detection log must refer to a registered user
    )
    
    first_seen : Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    last_seen : Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="present", nullable=False)
    # Relationships
    cctv: Mapped["CCTV"] = relationship("CCTV", back_populates="logs")
    lecture: Mapped["Lecture"] = relationship("Lecture", back_populates="logs")
