def score(ocr_confidence, anomaly_score, face_similarity):
    ocr = 0.5 if ocr_confidence is None else max(0, min(1, ocr_confidence))
    anomaly = 0 if anomaly_score is None else max(0, min(1, anomaly_score))
    risk = anomaly * 60
    reasons = []

    if anomaly > 0.4:
        reasons.append(f"Anomaly indicators detected (score={anomaly:.2f}).")
    risk += (1 - ocr) * 25
    if ocr < 0.6:
        reasons.append(f"Low OCR confidence ({ocr:.2f}).")

    if face_similarity is not None:
        face = max(0, min(1, face_similarity))
        risk += (1-face) * 15
        if face < 0.6:
            reasons.append(f"Low face similarity ({face:.2f}).")

    value = int(max(0, min(100, risk)))
    classification = "LOW_RISK" if value <= 30 else "REVIEW_REQUIRED" if value <= 70 else "HIGH_RISK"
    return {
        "risk_score": value,
        "classification": classification,
        "explanations": {
            "anomaly_score": anomaly,
            "ocr_confidence": ocr,
            "face_similarity": face_similarity,
            "reasons": reasons,
            "note": "Prototype heuristic score for hackathon demonstration; not a legally authoritative decision."
        }
    }
