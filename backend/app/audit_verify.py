from datetime import timezone

from .audit import AuditLog
from .audit_service import (
    _canonical_event_data,
    _calculate_event_hash,
)
from .database import SessionLocal


def verify_audit_chain():
    db = SessionLocal()

    try:
        rows = (
            db.query(AuditLog)
            .order_by(AuditLog.id.asc())
            .all()
        )

        print("=== SHIELDX AUDIT INTEGRITY CHECK ===")

        if not rows:
            print("No audit events found.")
            return False

        previous_hash = None
        overall_pass = True

        for row in rows:

            timestamp_utc = (
                row.timestamp
                .astimezone(timezone.utc)
                .isoformat()
            )

            canonical_data = _canonical_event_data(
                event_id=row.event_id,
                request_id=row.request_id,
                actor_id=row.actor_id,
                actor_role=row.actor_role,
                action=row.action,
                result=row.result,
                document_hash=row.document_hash,
                details=row.details,
                timestamp=timestamp_utc,
                previous_hash=row.previous_hash,
            )

            expected_hash = _calculate_event_hash(
                canonical_data
            )

            hash_ok = expected_hash == row.event_hash
            chain_ok = row.previous_hash == previous_hash

            print(
                f"EVENT {row.id}: {row.action} | "
                f"HASH: {'PASS' if hash_ok else 'FAIL'} | "
                f"CHAIN: {'PASS' if chain_ok else 'FAIL'}"
            )

            if not hash_ok or not chain_ok:
                overall_pass = False

            previous_hash = row.event_hash

        print(
            "OVERALL:",
            "PASS" if overall_pass else "FAIL",
        )

        return overall_pass

    finally:
        db.close()


if __name__ == "__main__":
    verify_audit_chain()