from typing import Any, Optional
from pydantic import BaseModel


class DocumentInfo(BaseModel):
    ocr_confidence: Optional[float] = None
    extracted_fields: dict[str, Any] = {}
    raw_text: str = ""


class FaceVerification(BaseModel):
    performed: bool
    similarity: Optional[float] = None
    notes: Optional[str] = None


class CrossDocumentVerification(BaseModel):
    performed: bool
    score: Optional[int] = None
    classification: str = "Not Submitted"
    matches: list[str] = []
    mismatches: list[str] = []


class VerificationResponse(BaseModel):
    request_id: str
    status: str
    risk_score: int
    classification: str
    document: DocumentInfo
    face_verification: FaceVerification
    cross_document: CrossDocumentVerification
    anomalies: list[dict[str, Any]]
    explanations: dict[str, Any]


class HistoryItem(BaseModel):
    request_id: str
    filename: str
    risk_score: int
    classification: str
    created_at: str
# ============================================================
# AUTHENTICATION
# ============================================================

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict[str, Any]
class LearningSampleReviewRequest(BaseModel):
    label: str
    notes: str | None = None