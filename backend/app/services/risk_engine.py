def score(
    ocr_confidence,
    anomaly_score,
    face_similarity,
    cross_document=None,
):
    ocr = 0.5 if ocr_confidence is None else max(0, min(1, ocr_confidence))
    anomaly = 0 if anomaly_score is None else max(0, min(1, anomaly_score))

    risk = anomaly * 60
    reasons = []

    if anomaly > 0.4:
        reasons.append(
            f"Anomaly indicators detected (score={anomaly:.2f})."
        )

    risk += (1 - ocr) * 25

    if ocr < 0.6:
        reasons.append(
            f"Low OCR confidence ({ocr:.2f})."
        )

    # Face verification risk.
    if face_similarity is not None:
        face = max(0, min(1, face_similarity))

        if face < 0.4:
            risk += 50
            reasons.append(
                f"Very low face similarity ({face:.2f})."
            )
        elif face < 0.6:
            risk += 30
            reasons.append(
                f"Low face similarity ({face:.2f})."
            )
        elif face < 0.75:
            risk += 15
            reasons.append(
                f"Moderate face similarity ({face:.2f})."
            )

    # Cross-document identity consistency risk.
    if cross_document and cross_document.get("performed"):
        mismatches = cross_document.get("mismatches", [])
        mismatch_count = len(mismatches)

        cross_document_risk = min(60, mismatch_count * 15)
        risk += cross_document_risk

        if mismatch_count:
            reasons.append(
                f"Cross-document verification found "
                f"{mismatch_count} mismatch(es)."
            )

            for mismatch in mismatches[:4]:
                reasons.append(mismatch)

    value = int(max(0, min(100, risk)))

    classification = (
        "LOW_RISK"
        if value <= 30
        else "REVIEW_REQUIRED"
        if value <= 70
        else "HIGH_RISK"
    )

    return {
        "risk_score": value,
        "classification": classification,
        "explanations": {
            "anomaly_score": anomaly,
            "ocr_confidence": ocr,
            "face_similarity": face_similarity,
            "cross_document": cross_document,
            "reasons": reasons,
            "note": (
                "Prototype heuristic score for hackathon demonstration; "
                "not a legally authoritative decision."
            ),
        },
    }