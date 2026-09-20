from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from ..models import LearningSample


MINIMUM_SAMPLES = 10
MINIMUM_CLASSES = 2
MINIMUM_SAMPLES_PER_CLASS = 2


@dataclass
class TrainingDataset:
    """Prepared dataset containing validated learning samples."""

    samples: list[LearningSample]

    @property
    def size(self) -> int:
        return len(self.samples)


class TrainingDataError(Exception):
    """Raised when the training dataset is not suitable for training."""


def validate_feature_label_consistency(
    samples: list[LearningSample],
) -> None:
    """
    Reject training data when identical feature vectors have
    contradictory reviewer labels.
    """

    feature_labels: dict[
        tuple[float, float, float, float],
        set[str],
    ] = {}

    for sample in samples:
        feature_vector = (
            float(sample.risk_score),
            float(
                sample.ocr_confidence
                if sample.ocr_confidence is not None
                else 0.0
            ),
            float(
                sample.face_similarity
                if sample.face_similarity is not None
                else 0.0
            ),
            float(
                sample.anomaly_score
                if sample.anomaly_score is not None
                else 0.0
            ),
        )

        label = sample.reviewer_label

        if label is None:
            continue

        feature_labels.setdefault(
            feature_vector,
            set(),
        ).add(label)

    contradictory_groups = [
        (features, labels)
        for features, labels in feature_labels.items()
        if len(labels) > 1
    ]

    if contradictory_groups:
        details = "; ".join(
            (
                f"features={features}, "
                f"labels={sorted(labels)}"
            )
            for features, labels in contradictory_groups[:5]
        )

        raise TrainingDataError(
            "Training data contains contradictory labels "
            "for identical feature vectors. "
            f"Conflicting groups: {details}"
        )


def build_training_dataset(
    db: Session,
    minimum_samples: int = MINIMUM_SAMPLES,
) -> TrainingDataset:
    """
    Collect reviewed, training-eligible samples that have not
    previously been used for training.
    """

    samples = (
        db.query(LearningSample)
        .filter(
            LearningSample.is_reviewed.is_(True),
            LearningSample.is_training_eligible.is_(True),
            LearningSample.reviewer_label.isnot(None),
            LearningSample.used_for_training.is_(False),
        )
        .order_by(
            LearningSample.id.asc()
        )
        .all()
    )

    if len(samples) < minimum_samples:
        raise TrainingDataError(
            "Insufficient training data. "
            f"Found {len(samples)} eligible samples, "
            f"but at least {minimum_samples} are required."
        )

    labels = [
        sample.reviewer_label
        for sample in samples
    ]

    distinct_labels = set(labels)

    if len(distinct_labels) < MINIMUM_CLASSES:
        raise TrainingDataError(
            "Insufficient label diversity. "
            f"Found {len(distinct_labels)} distinct label(s), "
            f"but at least {MINIMUM_CLASSES} are required."
        )

    label_counts = {}

    for label in labels:
        label_counts[label] = (
            label_counts.get(label, 0) + 1
        )

    insufficient_classes = [
        label
        for label, count in label_counts.items()
        if count < MINIMUM_SAMPLES_PER_CLASS
    ]

    if insufficient_classes:
        raise TrainingDataError(
            "Insufficient samples per class. "
            f"Classes requiring more samples: "
            f"{', '.join(insufficient_classes)}."
        )
    validate_feature_label_consistency(
        samples
    )

    return TrainingDataset(
        samples=samples
    )


def extract_features(
    samples: list[LearningSample],
) -> list[list[float]]:
    """
    Convert LearningSample records into numerical ML features.

    Feature order:
        1. risk_score
        2. ocr_confidence
        3. face_similarity
        4. anomaly_score
    """

    features = []

    for sample in samples:
        features.append(
            [
                float(sample.risk_score),
                float(
                    sample.ocr_confidence
                    if sample.ocr_confidence is not None
                    else 0.0
                ),
                float(
                    sample.face_similarity
                    if sample.face_similarity is not None
                    else 0.0
                ),
                float(
                    sample.anomaly_score
                    if sample.anomaly_score is not None
                    else 0.0
                ),
            ]
        )

    return features


def train_model(
    dataset: TrainingDataset,
) -> dict:
    """
    Train and validate a Random Forest classifier.

    This function performs another safety check before training.
    """

    samples = dataset.samples

    labels = [
        sample.reviewer_label
        for sample in samples
    ]

    features = extract_features(samples)

    if len(set(labels)) < MINIMUM_CLASSES:
        raise TrainingDataError(
            "Training requires at least two distinct classes."
        )

    try:
        (
            X_train,
            X_validation,
            y_train,
            y_validation,
        ) = train_test_split(
            features,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )

    except ValueError as exc:
        raise TrainingDataError(
            f"Unable to create a valid training/validation split: {exc}"
        ) from exc

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_validation
    )

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    precision = precision_score(
        y_validation,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "model": model,
        "training_samples": len(X_train),
        "validation_samples": len(X_validation),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "labels": sorted(
            set(labels)
        ),
    }