from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AuthorizedSourceResult:
    provider: str
    status: str
    verified: bool
    reference: str | None = None
    message: str | None = None
    data: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "status": self.status,
            "verified": self.verified,
            "reference": self.reference,
            "message": self.message,
            "data": self.data or {},
        }


class AuthorizedSourceService:
    """Safe abstraction for authorized-source verification."""

    def verify_document(
        self,
        *,
        document_type: str | None = None,
        document_number: str | None = None,
        name: str | None = None,
        dob: str | None = None,
    ) -> dict[str, Any]:

        result = AuthorizedSourceResult(
            provider="DigiLocker",
            status="NOT_CONFIGURED",
            verified=False,
            reference=None,
            message=(
                "DigiLocker authorized-source verification is not "
                "configured. No government-source verification was performed."
            ),
            data={
                "document_type": document_type,
                "document_number_provided": bool(document_number),
                "name_provided": bool(name),
                "dob_provided": bool(dob),
            },
        )

        return result.as_dict()


authorized_source_service = AuthorizedSourceService()