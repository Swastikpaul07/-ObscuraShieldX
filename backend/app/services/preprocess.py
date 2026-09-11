from pathlib import Path
from typing import Any
import cv2
import numpy as np
from PIL import Image

def pdf_first_page(path: str) -> str:
    from pdf2image import convert_from_path
    images = convert_from_path(path, first_page=1, last_page=1, dpi=180)
    if not images:
        raise ValueError("Could not render the PDF.")
    out = str(Path(path).with_suffix(".png"))
    images[0].convert("RGB").save(out)
    return out

def load_image_from_path(path: str):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    img_p = Image.open(path).convert("RGB")
    return np.array(img_p)[:, :, ::-1].copy()

def resize_keep_aspect(img, max_width=1400):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)

def denoise(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def detect_document_corners_and_warp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 75, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            rect = order_points(approx.reshape(4, 2))
            tl, tr, br, bl = rect
            max_w = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))
            max_h = max(int(np.linalg.norm(tr-br)), int(np.linalg.norm(tl-bl)))
            if max_w < 100 or max_h < 100:
                continue
            dst = np.array([[0,0],[max_w-1,0],[max_w-1,max_h-1],[0,max_h-1]], dtype="float32")
            matrix = cv2.getPerspectiveTransform(rect, dst)
            warp = cv2.warpPerspective(img, matrix, (max_w, max_h))
            return {"image": img, "warped": warp, "found": True}
    return {"image": img, "warped": None, "found": False}

def preprocess_document(path: str) -> dict[str, Any]:
    img = resize_keep_aspect(load_image_from_path(path))
    img = enhance_contrast(denoise(img))
    warp = detect_document_corners_and_warp(img)
    final = warp["warped"] if warp["warped"] is not None else img
    return {"image": final, "debug": {"warp_found": warp["found"]}}
