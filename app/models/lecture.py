import datetime
from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from app.database.database import Base
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.detection_log import DetectionLog

class Lecture(Base):
    __tablename__ = "lectures"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    nis: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    gender: Mapped[str] = mapped_column(String(50), nullable=False)
    jabatan: Mapped[str] = mapped_column(String(100), nullable=True)
    program_studi: Mapped[str] = mapped_column(String(150), nullable=True)
    jabatan_struktural: Mapped[str] = mapped_column(String(150), nullable=True)
    images: Mapped[str] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Relationships
    logs: Mapped[list["DetectionLog"]] = relationship(
        "DetectionLog",
        back_populates="lecture",
        cascade="all, delete-orphan"
    )
