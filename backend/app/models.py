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
    document_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
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

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=now,
        onupdate=now,
    )
class LearningSample(Base):
    __tablename__ = "learning_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    screening_id: Mapped[int] = mapped_column(
        ForeignKey("screenings.id"),
        nullable=False,
        index=True,
    )

    # Model/inference information captured at screening time
    risk_score: Mapped[int] = mapped_column(Integer, nullable=False)
    classification: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    ocr_confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    face_similarity: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    anomaly_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    # Reviewer-provided ground truth
    reviewer_label: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
    )

    reviewer_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
    )

    reviewer_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Learning lifecycle
    is_reviewed: Mapped[bool] = mapped_column(
        default=False,
        nullable=False,
    )

    is_training_eligible: Mapped[bool] = mapped_column(
        default=False,
        nullable=False,
    )

    used_for_training: Mapped[bool] = mapped_column(
        default=False,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=now,
    )

    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )
class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    version: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
    )

    model_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="candidate",
    )

    training_samples: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    validation_accuracy: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    validation_precision: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    validation_recall: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    validation_f1: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=now,
    )

    activated_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    artifact_path: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
    )