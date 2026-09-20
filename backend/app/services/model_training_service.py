from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
from sqlalchemy.orm import Session

from ..models import ModelVersion
from .training_service import (
    TrainingDataError,
    build_training_dataset,
    validate_feature_label_consistency,
    train_model,
)


# ---------------------------------------------------------
# MODEL ARTIFACT STORAGE
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"


def create_model_candidate(
    db: Session,
) -> dict:
    """
    Train a candidate model from reviewed learning samples.

    The candidate is recorded in ModelVersion but is NOT activated.

    The trained model is saved as a .joblib artifact and the
    artifact path is stored in ModelVersion.
    """

    # ---------------------------------------------------------
    # BUILD AND VALIDATE DATASET
    # ---------------------------------------------------------

    dataset = build_training_dataset(db)

    # ---------------------------------------------------------
    # VALIDATE FEATURE-LABEL CONSISTENCY
    # ---------------------------------------------------------

    validate_feature_label_consistency(
        dataset.samples
    )

    # ---------------------------------------------------------
    # TRAIN + VALIDATE
    # ---------------------------------------------------------
    training_result = train_model(
        dataset
    )

    # ---------------------------------------------------------
    # GENERATE VERSION
    # ---------------------------------------------------------

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d%H%M%S")

    version = (
        f"shieldx-{timestamp}"
    )

    # ---------------------------------------------------------
    # CREATE MODEL DIRECTORY
    # ---------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # SAVE TRAINED MODEL ARTIFACT
    # ---------------------------------------------------------

    artifact_file = (
        MODEL_DIR / f"{version}.joblib"
    )

    joblib.dump(
        training_result["model"],
        artifact_file,
    )

    # ---------------------------------------------------------
    # STORE MODEL VERSION
    # ---------------------------------------------------------

    model_version = ModelVersion(
        version=version,
        model_type="risk_classifier",
        artifact_path=str(
            artifact_file
        ),
        status="candidate",
        training_samples=training_result[
            "training_samples"
        ],
        validation_accuracy=training_result[
            "accuracy"
        ],
        validation_precision=training_result[
            "precision"
        ],
        validation_recall=training_result[
            "recall"
        ],
        validation_f1=training_result[
            "f1"
        ],
    )

    db.add(
        model_version
    )

    db.flush()

    # ---------------------------------------------------------
    # MARK SAMPLES AS USED FOR TRAINING
    # ---------------------------------------------------------

    for sample in dataset.samples:

        sample.used_for_training = True

    db.flush()

    # ---------------------------------------------------------
    # RETURN METADATA ONLY
    # ---------------------------------------------------------

    return {
        "model_version_id": model_version.id,
        "version": model_version.version,
        "status": model_version.status,
        "model_type": model_version.model_type,
        "artifact_path": model_version.artifact_path,
        "training_samples": model_version.training_samples,
        "validation_samples": training_result[
            "validation_samples"
        ],
        "validation_accuracy": model_version.validation_accuracy,
        "validation_precision": model_version.validation_precision,
        "validation_recall": model_version.validation_recall,
        "validation_f1": model_version.validation_f1,
        "labels": training_result[
            "labels"
        ],
    }