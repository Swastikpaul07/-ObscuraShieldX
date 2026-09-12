import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Screening, RiskFlag
from ..schemas.schemas import VerificationResponse, HistoryItem
from ..services import (
    preprocess,
    ocr_service,
    anomaly_service,
    face_service,
    risk_engine,
    hash_service,
)
from ..utils.file_utils import allowed_file, save_upload_file_tmp

router = APIRouter()
BASE_DIR = Path(__file__).resolve().parents[3]
UPLOAD_ROOT = BASE_DIR / "uploads"

@router.get("/", response_class=HTMLResponse)
async def index():
    return "<h1>ShieldX</h1><p>Open the ShieldX dashboard in your browser.</p>"

@router.get("/health")
async def health():
    return {"status": "ok", "service": "shieldx-api"}

@router.post(
    "/api/v1/verify/document",
    response_model=VerificationResponse,
)
async def verify_document(
    document: UploadFile = File(...),
    selfie: UploadFile | None = File(None),
    db: Session = Depends(get_db),
):
    request_id = str(uuid.uuid4())

    if not document.filename or not allowed_file(document.filename):
        raise HTTPException(status_code=400, detail="Unsupported document file type.")

    if selfie and (not selfie.filename or not allowed_file(selfie.filename, selfie=True)):
        raise HTTPException(status_code=400, detail="Unsupported selfie file type.")

    temp_dir = UPLOAD_ROOT / request_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        doc_path = await save_upload_file_tmp(document, str(temp_dir))
        selfie_path = None
        if selfie:
            selfie_path = await save_upload_file_tmp(selfie, str(temp_dir))

        # PDF support: convert first page when possible.
        document_hash = hash_service.calculate_sha256(doc_path)

        process_path = doc_path
        if Path(doc_path).suffix.lower() == ".pdf":
            process_path = preprocess.pdf_first_page(doc_path)

        prepped_doc = preprocess.preprocess_document(process_path)
        ocr_result = ocr_service.hybrid_ocr(
            prepped_doc["image"],
             source_path=doc_path,
        )
        anomalies = anomaly_service.analyze_document(
            prepped_doc["image"], source_path=doc_path
        )

        face_result = {
            "performed": False,
            "similarity": None,
            "notes": "Selfie not provided.",
        }
        if selfie_path:
            face_result = face_service.compare_faces(
                prepped_doc["image"], selfie_path
            )

        risk = risk_engine.score(
            ocr_confidence=ocr_result.get("avg_confidence"),
            anomaly_score=anomalies.get("anomaly_score"),
            face_similarity=face_result.get("similarity"),
        )

        result = {
            "request_id": request_id,
            "status": "completed",
            "risk_score": risk["risk_score"],
            "classification": risk["classification"],
            "document": {
                "ocr_confidence": ocr_result.get("avg_confidence"),
                "extracted_fields": ocr_result.get("fields", {}),
                "raw_text": ocr_result.get("text", ""),
            },
            "face_verification": {
                "performed": face_result.get("performed", False),
                "similarity": face_result.get("similarity"),
                "notes": face_result.get("notes"),
            },
            "anomalies": anomalies.get("indicators", []),
            "explanations": risk["explanations"],
        }

        screening = Screening(
            request_id=request_id,
            filename=document.filename,
            risk_score=risk["risk_score"],
            classification=risk["classification"],
            document_hash=document_hash,
            ocr_confidence=ocr_result.get("avg_confidence"),
            face_similarity=face_result.get("similarity"),
            anomaly_score=anomalies.get("anomaly_score"),
        )
        db.add(screening)
        db.flush()

        for item in anomalies.get("indicators", []):
            db.add(
                RiskFlag(
                    screening_id=screening.id,
                    flag_type=item.get("type", "anomaly"),
                    severity="high" if item.get("score", 0) >= 0.7 else "medium",
                    description=item.get("explanation", "Anomaly indicator detected."),
                )
            )
        for reason in risk["explanations"].get("reasons", []):
            db.add(
                RiskFlag(
                    screening_id=screening.id,
                    flag_type="risk_reason",
                    severity="high" if risk["risk_score"] > 70 else "medium",
                    description=reason,
                )
            )

        db.commit()
        return result

    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Processing failed: {exc}"
        ) from exc
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@router.get("/api/v1/screenings", response_model=list[HistoryItem])
def screenings(limit: int = 20, db: Session = Depends(get_db)):
    rows = (
        db.query(Screening)
        .order_by(Screening.created_at.desc())
        .limit(min(max(limit, 1), 100))
        .all()
    )
    return [
        HistoryItem(
            request_id=r.request_id,
            filename=r.filename,
            risk_score=r.risk_score,
            classification=r.classification,
            created_at=r.created_at.isoformat(),
        )
        for r in rows
    ]

@router.get("/api/v1/dashboard/stats")
def dashboard_stats(db: Session = Depends(get_db)):
    total = db.query(func.count(Screening.id)).scalar() or 0
    high = db.query(func.count(Screening.id)).filter(Screening.risk_score > 70).scalar() or 0
    review = db.query(func.count(Screening.id)).filter(
        Screening.risk_score.between(31, 70)
    ).scalar() or 0
    low = db.query(func.count(Screening.id)).filter(Screening.risk_score <= 30).scalar() or 0
    return {"total": total, "high_risk": high, "review_required": review, "low_risk": low}
