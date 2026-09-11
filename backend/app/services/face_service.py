import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def load(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

def detect_largest_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    x, y, w, h = sorted(faces, key=lambda r:r[2]*r[3], reverse=True)[0]
    return img[y:y+h, x:x+w]

def compare_faces(document_img, selfie_path):
    selfie = load(selfie_path)
    if selfie is None:
        return {"performed": False, "similarity": None, "notes": "Could not read selfie."}
    doc_face = detect_largest_face(document_img)
    selfie_face = detect_largest_face(selfie)
    if doc_face is None:
        return {"performed": False, "similarity": None, "notes": "No face detected in document."}
    if selfie_face is None:
        return {"performed": False, "similarity": None, "notes": "No face detected in selfie."}
    a = cv2.resize(cv2.cvtColor(doc_face, cv2.COLOR_BGR2GRAY), (160,160))
    b = cv2.resize(cv2.cvtColor(selfie_face, cv2.COLOR_BGR2GRAY), (160,160))
    sim, _ = ssim(a, b, full=True)
    return {"performed": True, "similarity": float(max(0,min(1,sim))), "notes": None}
