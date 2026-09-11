from datetime import datetime, timezone
from sqlalchemy import String, Integer, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .database import Base

def now():
    return datetime.now(timezone.utc)

class Screening(Base):
    __tablename__ = "screenings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    request_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    risk_score: Mapped[int] = mapped_column(Integer)
    classification: Mapped[str] = mapped_column(String(32))
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    face_similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)

    flags: Mapped[list["RiskFlag"]] = relationship(
        back_populates="screening", cascade="all, delete-orphan"
    )

class RiskFlag(Base):
    __tablename__ = "risk_flags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    screening_id: Mapped[int] = mapped_column(ForeignKey("screenings.id"))
    flag_type: Mapped[str] = mapped_column(String(80))
    severity: Mapped[str] = mapped_column(String(20))
    description: Mapped[str] = mapped_column(Text)

    screening: Mapped[Screening] = relationship(back_populates="flags")
