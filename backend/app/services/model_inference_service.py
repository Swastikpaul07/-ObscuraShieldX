from __future__ import annotations

from pathlib import Path

import joblib
from sqlalchemy.orm import Session

from ..models import ModelVersion


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = (BASE_DIR / "models").resolve()


class ModelInferenceError(Exception):
    """Raised when an active model cannot be loaded safely."""


def get_active_model_version(
    db: Session,
) -> ModelVersion | None:
    """Return the currently active model version, if one exists."""

    return (
        db.query(ModelVersion)
        .filter(
            ModelVersion.status == "active"
        )
        .order_by(
            ModelVersion.activated_at.desc()
        )
        .first()
    )


def load_active_model(
    db: Session,
):
    """
    Load the currently active ShieldX model artifact.

    Returns:
        (model_version, model)

    Returns (None, None) when no model is active.
    """

    model_version = get_active_model_version(db)

    if model_version is None:
        return None, None

    if not model_version.artifact_path:
        raise ModelInferenceError(
            "Active model has no artifact path."
        )

    artifact_path = Path(
        model_version.artifact_path
    ).resolve()

    try:
        artifact_path.relative_to(
            MODEL_DIR
        )
    except ValueError as exc:
        raise ModelInferenceError(
            "Active model artifact is outside the "
            "ShieldX model directory."
        ) from exc

    if not artifact_path.exists():
        raise ModelInferenceError(
            "Active model artifact does not exist."
        )

    if not artifact_path.is_file():
        raise ModelInferenceError(
            "Active model artifact is not a file."
        )

    try:
        model = joblib.load(
            artifact_path
        )
    except Exception as exc:
        raise ModelInferenceError(
            f"Unable to load active model artifact: {exc}"
        ) from exc

    return model_version, model


def predict_with_active_model(
    db: Session,
    risk_score: int,
    ocr_confidence: float | None,
    face_similarity: float | None,
    anomaly_score: float | None,
) -> dict | None:
    """
    Predict the reviewer classification using the active model.

    Returns None when no model is currently active.
    """

    model_version, model = load_active_model(db)

    if model_version is None:
        return None

    features = [[
        float(risk_score),
        float(
            ocr_confidence
            if ocr_confidence is not None
            else 0.0
        ),
        float(
            face_similarity
            if face_similarity is not None
            else 0.0
        ),
        float(
            anomaly_score
            if anomaly_score is not None
            else 0.0
        ),
    ]]

    try:
        prediction = model.predict(
            features
        )
    except Exception as exc:
        raise ModelInferenceError(
            f"Active model prediction failed: {exc}"
        ) from exc

    if len(prediction) == 0:
        raise ModelInferenceError(
            "Active model returned no prediction."
        )

    return {
        "model_version_id": model_version.id,
        "version": model_version.version,
        "classification": str(
            prediction[0]
        ),
    }