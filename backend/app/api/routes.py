import joblib
import shutil
import uuid
from pathlib import Path
from datetime import datetime, timezone
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database import get_db
from ..rbac import (
    Role,
    get_current_role,
    get_current_user_id,
    require_permission,
)
from ..models import (
    Screening,
    RiskFlag,
    User,
    LearningSample,
    ModelVersion,
)
from ..schemas.schemas import (
    VerificationResponse,
    HistoryItem,
    LoginRequest,
    LoginResponse,
    LearningSampleReviewRequest,
)
from ..core.security import verify_password, create_access_token
from ..audit_service import create_audit_event
from ..services import (
    preprocess,
    ocr_service,
    anomaly_service,
    face_service,
    risk_engine,
    hash_service,
)
from ..services.model_training_service import (
    create_model_candidate,
)

from ..services.model_inference_service import (
    ModelInferenceError,
    predict_with_active_model,
)
from ..services.training_service import (
    TrainingDataError,
)
from ..utils.file_utils import (
    allowed_file,
    save_upload_file_tmp,
)


router = APIRouter()


# ============================================================
# AUTHENTICATION
# ============================================================

@router.post("/api/v1/auth/login", response_model=LoginResponse)
def login(
    credentials: LoginRequest,
    db: Session = Depends(get_db),
):
    user = (
        db.query(User)
        .filter(User.username == credentials.username)
        .first()
    )

    if not user or not user.is_active:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
        )

    if not verify_password(
        credentials.password,
        user.password_hash,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
        )

    access_token = create_access_token(
        user_id=user.id,
        username=user.username,
        role=user.role,
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "name": user.name,
            "role": user.role,
            "is_active": user.is_active,
        },
    }


BASE_DIR = Path(__file__).resolve().parents[3]
UPLOAD_ROOT = BASE_DIR / "uploads"


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_field(value):
    if value is None:
        return ""

    return " ".join(
        str(value)
        .lower()
        .strip()
        .split()
    )


# ============================================================
# APPLICANT + DOCUMENT + CROSS-DOCUMENT COMPARISON
# ============================================================

def compare_documents(
    primary_fields,
    supporting_fields=None,
    applicant_fields=None,
):
    """
    Compare:

    1. Applicant-entered information
       against the primary document.

    2. Primary document
       against the supporting document.
    """

    primary_fields = primary_fields or {}
    supporting_fields = supporting_fields or {}
    applicant_fields = applicant_fields or {}

    matches = []
    mismatches = []
    unavailable = []

    # --------------------------------------------------------
    # Applicant vs Primary Document
    # --------------------------------------------------------

    applicant_checks = [
        ("Applicant name", "name"),
        ("Applicant date of birth", "date_of_birth"),
        ("Applicant document number", "document_number"),
        ("Applicant address", "address"),
    ]

    for label, key in applicant_checks:

        applicant_value = normalize_field(
            applicant_fields.get(key)
        )

        document_value = normalize_field(
            primary_fields.get(key)
        )

        # Applicant did not provide this field.
        if not applicant_value:
            continue

        # OCR could not extract this field.
        if not document_value:
            unavailable.append(
                f"{label} — document field unavailable"
            )
            continue

        # Values match.
        if applicant_value == document_value:
            matches.append(
                f"{label} matches document"
            )

        # Values do not match.
        else:
            mismatches.append(
                f"{label} mismatch — "
                f"applicant='{applicant_value}' "
                f"vs document='{document_value}'"
            )

    # --------------------------------------------------------
    # Primary Document vs Supporting Document
    # --------------------------------------------------------

    if supporting_fields:

        document_checks = [
            ("Name consistency", "name"),
            ("Date of birth consistency", "date_of_birth"),
            ("Document number linkage", "document_number"),
            ("Address consistency", "address"),
        ]

        for label, key in document_checks:

            primary_value = normalize_field(
                primary_fields.get(key)
            )

            supporting_value = normalize_field(
                supporting_fields.get(key)
            )

            # One or both fields unavailable.
            if not primary_value or not supporting_value:
                unavailable.append(
                    f"{label} — field unavailable"
                )
                continue

            # Values match.
            if primary_value == supporting_value:
                matches.append(label)

            # Values do not match.
            else:
                mismatches.append(
                    f"{label} mismatch — "
                    f"primary='{primary_value}' "
                    f"vs supporting='{supporting_value}'"
                )

    # --------------------------------------------------------
    # Verification score
    # --------------------------------------------------------

    total_checks = (
        len(matches)
        + len(mismatches)
    )

    if total_checks == 0:

        score = 0

        classification = "Unable to Verify"

    else:

        score = round(
            (len(matches) / total_checks) * 100
        )

        if score >= 75:
            classification = "Verified"

        elif score >= 50:
            classification = "Review Required"

        else:
            classification = "Mismatch Detected"

    return {
        "performed": True,
        "score": score,
        "classification": classification,
        "matches": matches,
        "mismatches": mismatches,
        "unavailable": unavailable,
    }


# ============================================================
# HEALTH
# ============================================================

@router.get("/health")
async def health():

    return {
        "status": "ok",
        "service": "shieldx-api",
    }


# ============================================================
# DOCUMENT VERIFICATION
# ============================================================

@router.post(
    "/api/v1/verify/document",
    response_model=VerificationResponse,
)
async def verify_document(

    document: UploadFile = File(...),

    selfie: UploadFile | None = File(None),

    supporting_doc: UploadFile | None = File(None),

    applicant_name: str | None = Form(None),

    applicant_dob: str | None = Form(None),

    applicant_id_number: str | None = Form(None),

    applicant_address: str | None = Form(None),

    db: Session = Depends(get_db),
    role: Role = Depends(require_permission("screen")),
):

    request_id = str(uuid.uuid4())

    # Validate applicant date of birth
    if applicant_dob:
        from datetime import date
        try:
            dob = date.fromisoformat(applicant_dob)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid date of birth."
            )

        if dob > date.today():
            raise HTTPException(
                status_code=400,
                detail="Date of birth cannot be in the future."
            )
        
    # --------------------------------------------------------
    # Validate primary document
    # --------------------------------------------------------

    if (
        not document.filename
        or not allowed_file(document.filename)
    ):
        raise HTTPException(
            status_code=400,
            detail="Unsupported document file type.",
        )

    # --------------------------------------------------------
    # Validate selfie
    # --------------------------------------------------------

    if selfie and (
        not selfie.filename
        or not allowed_file(
            selfie.filename,
            selfie=True,
        )
    ):
        raise HTTPException(
            status_code=400,
            detail="Unsupported selfie file type.",
        )

    # --------------------------------------------------------
    # Validate supporting document
    # --------------------------------------------------------

    if supporting_doc and (
        not supporting_doc.filename
        or not allowed_file(
            supporting_doc.filename
        )
    ):
        raise HTTPException(
            status_code=400,
            detail="Unsupported supporting document file type.",
        )

    # --------------------------------------------------------
    # Temporary request directory
    # --------------------------------------------------------

    temp_dir = UPLOAD_ROOT / request_id

    temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        # ====================================================
        # SAVE PRIMARY DOCUMENT
        # ====================================================

        doc_path = await save_upload_file_tmp(
            document,
            str(temp_dir),
        )

        # ====================================================
        # SAVE SELFIE
        # ====================================================

        selfie_path = None

        if selfie:

            selfie_path = await save_upload_file_tmp(
                selfie,
                str(temp_dir),
            )

        # ====================================================
        # SAVE SUPPORTING DOCUMENT
        # ====================================================

        supporting_doc_path = None

        if supporting_doc:

            supporting_doc_path = (
                await save_upload_file_tmp(
                    supporting_doc,
                    str(temp_dir),
                )
            )

        # ====================================================
        # DOCUMENT HASH
        # ====================================================

        document_hash = (
            hash_service.calculate_sha256(
                doc_path
            )
        )

        # ====================================================
        # PREPARE PRIMARY DOCUMENT
        # ====================================================

        process_path = doc_path

        if (
            Path(doc_path)
            .suffix
            .lower()
            == ".pdf"
        ):

            process_path = (
                preprocess.pdf_to_image(
                    doc_path
                )
            )

        prepped_doc = (
            preprocess.preprocess_document(
                process_path
            )
        )

        # ====================================================
        # OCR PRIMARY DOCUMENT
        # ====================================================

        ocr_result = (
            ocr_service.hybrid_ocr(
                prepped_doc["image"],
                source_path=doc_path,
            )
        )

        # ====================================================
        # APPLICANT DATA
        # ====================================================

        applicant_fields = {

            "name": applicant_name,

            "date_of_birth": applicant_dob,

            "document_number": applicant_id_number,

            "address": applicant_address,
        }

        # ====================================================
        # CROSS-DOCUMENT VERIFICATION
        # ====================================================

        cross_document = {
            "performed": False,
            "score": None,
            "classification": "Not Submitted",
            "matches": [],
            "mismatches": [],
            "unavailable": [],
        }

        # ----------------------------------------------------
        # WITH SUPPORTING DOCUMENT
        # ----------------------------------------------------

        if supporting_doc_path:

            supporting_image = (
                supporting_doc_path
            )

            if (
                Path(supporting_doc_path)
                .suffix
                .lower()
                == ".pdf"
            ):

                supporting_image = (
                    preprocess.pdf_to_image(
                        supporting_doc_path
                    )
                )

            prepped_supporting = (
                preprocess.preprocess_document(
                    supporting_image
                )
            )

            supporting_ocr = (
                ocr_service.hybrid_ocr(
                    prepped_supporting["image"],
                    source_path=supporting_doc_path,
                )
            )

            cross_document = compare_documents(

                primary_fields=ocr_result.get(
                    "fields",
                    {},
                ),

                supporting_fields=supporting_ocr.get(
                    "fields",
                    {},
                ),

                applicant_fields=applicant_fields,
            )

        # ----------------------------------------------------
        # WITHOUT SUPPORTING DOCUMENT
        # ----------------------------------------------------

        else:

            cross_document = compare_documents(

                primary_fields=ocr_result.get(
                    "fields",
                    {},
                ),

                supporting_fields=None,

                applicant_fields=applicant_fields,
            )

        # ====================================================
        # ANOMALY / TAMPER ANALYSIS
        # ====================================================

        anomalies = (
            anomaly_service.analyze_document(
                prepped_doc["image"],
                source_path=doc_path,
            )
        )

        # ====================================================
        # FACE VERIFICATION
        # ====================================================

        face_result = {
            "performed": False,
            "similarity": None,
            "notes": "Selfie not provided.",
        }

        if selfie_path:

            face_result = (
                face_service.compare_faces(
                    prepped_doc["image"],
                    selfie_path,
                )
            )

        # ====================================================
        # RISK FUSION
        # ====================================================

        risk = risk_engine.score(

            ocr_confidence=ocr_result.get(
                "avg_confidence"
            ),

            anomaly_score=anomalies.get(
                "anomaly_score"
            ),

            face_similarity=face_result.get(
                "similarity"
            ),

            cross_document=cross_document,
        )

        # ====================================================
        # ACTIVE ML MODEL INFERENCE
        # ====================================================

        learned_prediction = None

        try:
            learned_prediction = predict_with_active_model(
                db=db,
                risk_score=risk["risk_score"],
                ocr_confidence=ocr_result.get(
                    "avg_confidence"
                ),
                face_similarity=face_result.get(
                    "similarity"
                ),
                anomaly_score=anomalies.get(
                    "anomaly_score"
                ),
            )

        except ModelInferenceError as exc:
            # The learned model is an optional enhancement.
            # Never allow a model-loading/prediction problem
            # to break the primary ShieldX screening pipeline.
            learned_prediction = {
                "status": "unavailable",
                "reason": str(exc),
            }

        # ====================================================
        # API RESULT
        # ====================================================

        result = {

            "request_id": request_id,

            "status": "completed",

            "risk_score": risk[
                "risk_score"
            ],

            "classification": risk[
                "classification"
            ],

            "learned_model": learned_prediction,

            "document": {

                "ocr_confidence": ocr_result.get(
                    "avg_confidence"
                ),

                "extracted_fields": ocr_result.get(
                    "fields",
                    {},
                ),

                "raw_text": ocr_result.get(
                    "text",
                    "",
                ),
            },

            "face_verification": {

                "performed": face_result.get(
                    "performed",
                    False,
                ),

                "similarity": face_result.get(
                    "similarity"
                ),

                "notes": face_result.get(
                    "notes"
                ),
            },

            "cross_document": cross_document,

            "anomalies": anomalies.get(
                "indicators",
                [],
            ),

            "explanations": risk[
                "explanations"
            ],
        }

        # ====================================================
        # DATABASE SCREENING RECORD
        # ====================================================

        screening = Screening(

            request_id=request_id,

            filename=document.filename,

            risk_score=risk[
                "risk_score"
            ],

            classification=risk[
                "classification"
            ],

            document_hash=document_hash,

            ocr_confidence=ocr_result.get(
                "avg_confidence"
            ),

            face_similarity=face_result.get(
                "similarity"
            ),

            anomaly_score=anomalies.get(
                "anomaly_score"
            ),
        )

        db.add(screening)

        db.flush()

        # ====================================================
        # CONTINUOUS LEARNING - PRIVATE SAMPLE CAPTURE
        # ====================================================

        learning_sample = LearningSample(
            screening_id=screening.id,
            risk_score=risk["risk_score"],
            classification=risk["classification"],
            ocr_confidence=ocr_result.get(
                "avg_confidence"
            ),
            face_similarity=face_result.get(
                "similarity"
            ),
            anomaly_score=anomalies.get(
                "anomaly_score"
            ),
        )

        db.add(learning_sample)

        # ====================================================
        # ANOMALY RISK FLAGS
        # ====================================================

        for item in anomalies.get(
            "indicators",
            [],
        ):

            db.add(
                RiskFlag(

                    screening_id=screening.id,

                    flag_type=item.get(
                        "type",
                        "anomaly",
                    ),

                    severity=(
                        "high"
                        if item.get(
                            "score",
                            0,
                        ) >= 0.7
                        else "medium"
                    ),

                    description=item.get(
                        "explanation",
                        "Anomaly indicator detected.",
                    ),
                )
            )

        # ====================================================
        # RISK EXPLANATION FLAGS
        # ====================================================

        for reason in risk[
            "explanations"
        ].get(
            "reasons",
            [],
        ):

            db.add(
                RiskFlag(

                    screening_id=screening.id,

                    flag_type="risk_reason",

                    severity=(
                        "high"
                        if risk[
                            "risk_score"
                        ] > 70
                        else "medium"
                    ),

                    description=reason,
                )
            )

        # ====================================================
        # AUDIT LOG
        # ====================================================

        create_audit_event(
            db,
            action="DOCUMENT_SCREENING",
            result=risk["classification"],
            request_id=request_id,
            actor_id="system",
            actor_role="SYSTEM",
            document_hash=document_hash,
            details=(
                f"risk_score={risk['risk_score']};"
                f"classification={risk['classification']}"
            ),
        )

        # ====================================================
        # COMMIT
        # ====================================================

        db.commit()

        return result
        return result

    # ========================================================
    # HTTP EXCEPTION
    # ========================================================

    except HTTPException:

        raise

    # ========================================================
    # GENERAL ERROR
    # ========================================================

    except Exception as exc:

        db.rollback()

        raise HTTPException(

            status_code=500,

            detail=(
                f"Processing failed: {exc}"
            ),

        ) from exc

    # ========================================================
    # CLEAN TEMPORARY FILES
    # ========================================================

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )


# ============================================================
# SCREENING HISTORY
# ============================================================

@router.get(
    "/api/v1/screenings",
    response_model=list[HistoryItem],
)
def screenings(

    limit: int = 20,

    db: Session = Depends(get_db),

     role: Role = Depends(require_permission("review")),

):

    rows = (
        db.query(Screening)
        .order_by(
            Screening.created_at.desc()
        )
        .limit(
            min(
                max(limit, 1),
                100,
            )
        )
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
# ============================================================
# SCREENING DETAIL
# ============================================================

@router.get(
    "/api/v1/screenings/{request_id}"
)
def screening_detail(
    request_id: str,
    db: Session = Depends(get_db),
    role: Role = Depends(require_permission("review")),
):
    screening = (
        db.query(Screening)
        .filter(
            Screening.request_id == request_id
        )
        .first()
    )

    if not screening:
        raise HTTPException(
            status_code=404,
            detail="Screening not found."
        )

    flags = (
        db.query(RiskFlag)
        .filter(
            RiskFlag.screening_id == screening.id
        )
        .all()
    )

    return {
        "request_id": screening.request_id,
        "status": "completed",
        "filename": screening.filename,
        "risk_score": screening.risk_score,
        "classification": screening.classification,
        "document_hash": screening.document_hash,
        "ocr_confidence": screening.ocr_confidence,
        "face_similarity": screening.face_similarity,
        "anomaly_score": screening.anomaly_score,
        "created_at": (
            screening.created_at.isoformat()
            if screening.created_at
            else None
        ),
        "risk_flags": [
            {
                "type": flag.flag_type,
                "severity": flag.severity,
                "description": flag.description,
            }
            for flag in flags
        ],
    }
# ============================================================
# LEARNING SAMPLE REVIEW
# ============================================================

@router.get(
    "/api/v1/learning-samples"
)
def learning_samples(
    db: Session = Depends(get_db),
    role: Role = Depends(require_permission("review")),
):
    """
    Return learning samples waiting for human review.
    """

    samples = (
        db.query(LearningSample)
        .filter(
            LearningSample.is_reviewed == False
        )
        .order_by(
            LearningSample.created_at.desc()
        )
        .all()
    )

    return [
        {
            "id": sample.id,
            "screening_id": sample.screening_id,
            "risk_score": sample.risk_score,
            "classification": sample.classification,
            "ocr_confidence": sample.ocr_confidence,
            "face_similarity": sample.face_similarity,
            "anomaly_score": sample.anomaly_score,
            "reviewer_label": sample.reviewer_label,
            "reviewer_notes": sample.reviewer_notes,
            "is_reviewed": sample.is_reviewed,
            "is_training_eligible": sample.is_training_eligible,
            "used_for_training": sample.used_for_training,
            "created_at": (
                sample.created_at.isoformat()
                if sample.created_at
                else None
            ),
        }
        for sample in samples
    ]


# ============================================================
# REVIEW A LEARNING SAMPLE
# ============================================================

@router.patch(
    "/api/v1/learning-samples/{sample_id}/review"
)
def review_learning_sample(
    sample_id: int,
    review_data: LearningSampleReviewRequest,
    db: Session = Depends(get_db),
    role: Role = Depends(require_permission("review")),
    reviewer_id: int = Depends(get_current_user_id),
):
    """
    Human reviewer evaluates a learning sample.

    Approved labels become eligible for future model training.
    """

    sample = (
        db.query(LearningSample)
        .filter(
            LearningSample.id == sample_id
        )
        .first()
    )
    reviewer = (
        db.query(User)
        .filter(
            User.id == reviewer_id,
            User.is_active.is_(True),
        )
        .first()
    )

    if not reviewer:
        raise HTTPException(
            status_code=401,
            detail="Authenticated reviewer account is not active.",
        )

    if not sample:
        raise HTTPException(
            status_code=404,
            detail="Learning sample not found.",
        )

    if sample.is_reviewed:
        raise HTTPException(
            status_code=400,
            detail="Learning sample has already been reviewed.",
        )

    # --------------------------------------------------------
    # Validate reviewer label
    # --------------------------------------------------------

    allowed_labels = {
        "VERIFIED",
        "HIGH_RISK",
        "MISMATCH_DETECTED",
        "REVIEW_REQUIRED",
    }

    label = review_data.label.strip().upper()

    if label not in allowed_labels:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid reviewer label. "
                "Allowed labels: "
                "VERIFIED, HIGH_RISK, "
                "MISMATCH_DETECTED, REVIEW_REQUIRED."
            ),
        )

    # --------------------------------------------------------
    # Update review information
    # --------------------------------------------------------

    sample.reviewer_label = label

    sample.reviewer_id = reviewer.id

    sample.reviewer_notes = (
        review_data.notes.strip()
        if review_data.notes
        else None
    )

    sample.is_reviewed = True

    sample.is_training_eligible = True

    sample.used_for_training = False

    sample.reviewed_at = datetime.now(timezone.utc)

    db.commit()

    db.refresh(sample)

    return {
        "message": "Learning sample reviewed successfully.",
        "sample_id": sample.id,
        "screening_id": sample.screening_id,
        "reviewer_label": sample.reviewer_label,
        "reviewer_notes": sample.reviewer_notes,
        "is_reviewed": sample.is_reviewed,
        "is_training_eligible": sample.is_training_eligible,
        "used_for_training": sample.used_for_training,
        "reviewed_at": (
            sample.reviewed_at.isoformat()
            if sample.reviewed_at
            else None
        ),
    }
# ============================================================
# DASHBOARD STATISTICS
# ============================================================

@router.get(
    "/api/v1/dashboard/stats"
)
def dashboard_stats(
    role: Role = Depends(require_permission("audit")),
    db: Session = Depends(get_db),
):

    total = (
        db.query(
            func.count(Screening.id)
        )
        .scalar()
        or 0
    )

    high = (
        db.query(
            func.count(Screening.id)
        )
        .filter(
            Screening.risk_score > 70
        )
        .scalar()
        or 0
    )

    review = (
        db.query(
            func.count(Screening.id)
        )
        .filter(
            Screening.risk_score.between(
                31,
                70,
            )
        )
        .scalar()
        or 0
    )

    low = (
        db.query(
            func.count(Screening.id)
        )
        .filter(
            Screening.risk_score <= 30
        )
        .scalar()
        or 0
    )

    return {

        "total": total,

        "high_risk": high,

        "review_required": review,

        "low_risk": low,
    }
# ============================================================
# MODEL TRAINING
# ============================================================

@router.post(
    "/api/v1/training/run"
)
def run_training(
    role: Role = Depends(
        require_permission("admin")
    ),
    db: Session = Depends(get_db),
):
    """
    Train a new candidate model from reviewed learning samples.

    Training is protected by the admin permission and will never
    automatically activate the resulting model.
    """

    try:

        result = create_model_candidate(
            db
        )

        # Do not expose the actual sklearn model object
        # through the API response.
        result.pop(
            "model",
            None,
        )

        db.commit()

        return {
            "status": "candidate_created",
            "message": (
                "Training completed successfully. "
                "The model remains a candidate and "
                "has not been activated."
            ),
            **result,
        }

    except TrainingDataError as exc:

        db.rollback()

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=(
                f"Training failed: {exc}"
            ),
        ) from exc
    # ============================================================
    # MODEL ACTIVATION
    # ============================================================

@router.post(
    "/api/v1/training/models/{model_version_id}/activate"
)
def activate_model(
    model_version_id: int,
    role: Role = Depends(
        require_permission("admin")
    ),
    db: Session = Depends(get_db),
):
    """
    Activate an approved candidate model.

    Only administrators can activate a model.

    The currently active model is retired before the
    selected candidate becomes active.
    """

    # ---------------------------------------------------------
    # FIND CANDIDATE MODEL
    # ---------------------------------------------------------

    model_version = (
        db.query(ModelVersion)
        .filter(
            ModelVersion.id == model_version_id
        )
        .first()
    )

    if not model_version:
        raise HTTPException(
            status_code=404,
            detail="Model version not found.",
        )

    # ---------------------------------------------------------
    # VERIFY MODEL IS A CANDIDATE
    # ---------------------------------------------------------

    if model_version.status != "candidate":
        raise HTTPException(
            status_code=400,
            detail=(
                "Only candidate models can be activated."
            ),
        )

    # ---------------------------------------------------------
    # VERIFY ARTIFACT EXISTS
    # ---------------------------------------------------------

    if not model_version.artifact_path:
        raise HTTPException(
            status_code=400,
            detail=(
                "Model candidate has no artifact path."
            ),
        )

    artifact_path = Path(
        model_version.artifact_path
    ).resolve()

    # ---------------------------------------------------------
    # SECURITY: ARTIFACT MUST STAY INSIDE MODEL DIRECTORY
    # ---------------------------------------------------------

    model_directory = (
        Path(__file__)
        .resolve()
        .parents[2]
        / "models"
    ).resolve()

    try:
        artifact_path.relative_to(
            model_directory
        )
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=(
                "Model artifact is outside the "
                "ShieldX model directory."
            ),
        )

    # ---------------------------------------------------------
    # VERIFY ARTIFACT EXISTS AND IS A FILE
    # ---------------------------------------------------------

    if not artifact_path.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                "Model artifact does not exist."
            ),
        )

    if not artifact_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=(
                "Model artifact is not a file."
            ),
        )

    # ---------------------------------------------------------
    # VERIFY MODEL CAN ACTUALLY BE LOADED
    # ---------------------------------------------------------

    try:
        joblib.load(
            artifact_path
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Model artifact could not be loaded: "
                f"{exc}"
            ),
        ) from exc

    # ---------------------------------------------------------
    # RETIRE CURRENT ACTIVE MODEL
    # ---------------------------------------------------------

    active_models = (
        db.query(ModelVersion)
        .filter(
            ModelVersion.status == "active"
        )
        .all()
    )

    for active_model in active_models:

        active_model.status = "retired"

    # ---------------------------------------------------------
    # ACTIVATE SELECTED MODEL
    # ---------------------------------------------------------

    model_version.status = "active"

    model_version.activated_at = datetime.now(
        timezone.utc
    )

    db.commit()

    db.refresh(
        model_version
    )

    return {
        "status": "active",
        "message": (
            "Model activated successfully."
        ),
        "model_version_id": model_version.id,
        "version": model_version.version,
        "model_type": model_version.model_type,
        "artifact_path": model_version.artifact_path,
        "activated_at": model_version.activated_at,
    }
# ============================================================
# MODEL ROLLBACK
# ============================================================

@router.post(
    "/api/v1/training/models/{model_version_id}/rollback"
)
def rollback_model(
    model_version_id: int,
    role: Role = Depends(
        require_permission("admin")
    ),
    db: Session = Depends(get_db),
):
    """
    Roll back from the selected active model to the
    most recently activated retired model.

    Only administrators can perform rollback.
    """

    # ---------------------------------------------------------
    # FIND CURRENT MODEL
    # ---------------------------------------------------------

    current_model = (
        db.query(ModelVersion)
        .filter(
            ModelVersion.id == model_version_id
        )
        .first()
    )

    if not current_model:
        raise HTTPException(
            status_code=404,
            detail="Model version not found.",
        )

    # ---------------------------------------------------------
    # MODEL MUST CURRENTLY BE ACTIVE
    # ---------------------------------------------------------

    if current_model.status != "active":
        raise HTTPException(
            status_code=400,
            detail=(
                "Only the currently active model "
                "can be rolled back."
            ),
        )

    # ---------------------------------------------------------
    # FIND MOST RECENT RETIRED MODEL
    # ---------------------------------------------------------

    previous_model = (
        db.query(ModelVersion)
        .filter(
            ModelVersion.status == "retired",
            ModelVersion.activated_at.isnot(None),
        )
        .order_by(
            ModelVersion.activated_at.desc()
        )
        .first()
    )

    if not previous_model:
        raise HTTPException(
            status_code=400,
            detail=(
                "No previous retired model is available "
                "for rollback."
            ),
        )

    # ---------------------------------------------------------
    # VERIFY PREVIOUS ARTIFACT EXISTS
    # ---------------------------------------------------------

    if not previous_model.artifact_path:
        raise HTTPException(
            status_code=400,
            detail=(
                "Previous model has no artifact path."
            ),
        )

    previous_artifact = Path(
        previous_model.artifact_path
    )

    if not previous_artifact.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                "Previous model artifact does not exist."
            ),
        )

    # ---------------------------------------------------------
    # RETIRE CURRENT MODEL
    # ---------------------------------------------------------

    current_model.status = "retired"

    # ---------------------------------------------------------
    # RESTORE PREVIOUS MODEL
    # ---------------------------------------------------------

    previous_model.status = "active"

    previous_model.activated_at = datetime.now(
        timezone.utc
    )

    db.commit()

    db.refresh(
        previous_model
    )

    return {
        "status": "rollback_completed",
        "message": (
            "Model rollback completed successfully."
        ),
        "active_model_version_id": (
            previous_model.id
        ),
        "active_version": (
            previous_model.version
        ),
        "artifact_path": (
            previous_model.artifact_path
        ),
        "activated_at": (
            previous_model.activated_at
        ),
    }