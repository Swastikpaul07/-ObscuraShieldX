import os
import cv2
import numpy as np
from PIL import Image

def image_entropy(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / max(hist.sum(), 1)
    return float(-np.sum([p * np.log2(p) for p in hist if p > 0]))

def check_blur(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def has_suspicious_metadata(path):
    try:
        info = Image.open(path).info
        raw = str(info).lower()
        return ("photoshop" in raw or "adobe" in raw), info
    except Exception:
        return False, {}

def simple_duplicated_region_score(gray):
    small = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25)
    h, w = small.shape
    bh, bw = 24, 24
    blocks = [
        small[y:y+bh, x:x+bw]
        for y in range(0, h-bh+1, bh)
        for x in range(0, w-bw+1, bw)
    ][:80]
    if len(blocks) < 2:
        return 0.0
    matches = comparisons = 0
    for i in range(len(blocks)-1):
        a = blocks[i]
        for j in range(i+1, len(blocks)):
            b = blocks[j]
            comparisons += 1
            if a.std() == 0 or b.std() == 0:
                continue
            if np.corrcoef(a.flatten(), b.flatten())[0,1] > 0.92:
                matches += 1
    return float(matches / max(1, comparisons))

def analyze_document(img_cv, source_path=None):
    indicators = []
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = check_blur(gray)
    entropy = image_entropy(gray)

    if blur < 100:
        indicators.append({"type": "blurry", "score": min(1.0, 1-blur/100),
                           "explanation": f"Low sharpness detected (Laplacian variance {blur:.1f})."})
    if entropy < 4.0:
        indicators.append({"type": "low_entropy", "score": min(1.0, 1-entropy/4),
                           "explanation": f"Low image entropy ({entropy:.2f}) may indicate smoothing/compression."})

    meta_flag = False
    if source_path and os.path.exists(source_path) and not source_path.lower().endswith(".pdf"):
        meta_flag, _ = has_suspicious_metadata(source_path)
        if meta_flag:
            indicators.append({"type": "suspicious_metadata", "score": 0.7,
                               "explanation": "Image metadata contains editing-software indicators."})

    dup = simple_duplicated_region_score(gray)
    if dup > 0.15:
        indicators.append({"type": "duplicated_regions", "score": dup,
                           "explanation": f"Repeated image patterns detected (score {dup:.2f})."})

    b = min(1.0, max(0.0, 1-blur/300))
    e = min(1.0, max(0.0, 1-entropy/8))
    anomaly = max(0.0, min(1.0, 0.4*b + 0.2*e + 0.3*dup + 0.1*(0.7 if meta_flag else 0)))
    return {"anomaly_score": float(anomaly), "indicators": indicators}
