from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Integer, String, Text

from .database import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)

    event_id = Column(String(64), unique=True, nullable=False, index=True)
    request_id = Column(String(64), nullable=True, index=True)

    actor_id = Column(String(128), nullable=True, index=True)
    actor_role = Column(String(32), nullable=True, index=True)

    action = Column(String(128), nullable=False, index=True)
    result = Column(String(32), nullable=False)

    document_hash = Column(String(64), nullable=True, index=True)

    details = Column(Text, nullable=True)

    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    previous_hash = Column(String(64), nullable=True)
    event_hash = Column(String(64), nullable=False, index=True)