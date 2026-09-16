import hashlib
import json
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select

from .audit import AuditLog


def _canonical_event_data(
    event_id,
    request_id,
    actor_id,
    actor_role,
    action,
    result,
    document_hash,
    details,
    timestamp,
    previous_hash,
):
    """
    Create a deterministic representation of an audit event
    before calculating its integrity hash.
    """

    data = {
        "event_id": event_id,
        "request_id": request_id,
        "actor_id": actor_id,
        "actor_role": actor_role,
        "action": action,
        "result": result,
        "document_hash": document_hash,
        "details": details,
        "timestamp": timestamp,
        "previous_hash": previous_hash,
    }

    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _calculate_event_hash(canonical_data):
    return hashlib.sha256(canonical_data.encode("utf-8")).hexdigest()


def create_audit_event(
    db,
    *,
    action,
    result,
    request_id=None,
    actor_id=None,
    actor_role=None,
    document_hash=None,
    details=None,
):
    """
    Create a tamper-evident audit event.

    Sensitive document contents should NOT be stored here.
    Only metadata and the document hash should be recorded.
    """

    last_event = db.execute(
        select(AuditLog)
        .order_by(AuditLog.id.desc())
        .limit(1)
    ).scalar_one_or_none()

    previous_hash = last_event.event_hash if last_event else None

    event_id = uuid4().hex
    timestamp = datetime.now(timezone.utc)

    timestamp_value = timestamp.isoformat()

    canonical_data = _canonical_event_data(
        event_id=event_id,
        request_id=request_id,
        actor_id=actor_id,
        actor_role=actor_role,
        action=action,
        result=result,
        document_hash=document_hash,
        details=details,
        timestamp=timestamp_value,
        previous_hash=previous_hash,
    )

    event_hash = _calculate_event_hash(canonical_data)

    audit_event = AuditLog(
        event_id=event_id,
        request_id=request_id,
        actor_id=actor_id,
        actor_role=actor_role,
        action=action,
        result=result,
        document_hash=document_hash,
        details=details,
        timestamp=timestamp,
        previous_hash=previous_hash,
        event_hash=event_hash,
    )

    db.add(audit_event)
    db.flush()

    return audit_event