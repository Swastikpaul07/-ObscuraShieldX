import os
import secrets
import shutil
from pathlib import Path
from fastapi import UploadFile

DOCUMENT_EXT = {"jpg", "jpeg", "png", "pdf"}
SELFIE_EXT = {"jpg", "jpeg", "png"}
MAX_FILE_SIZE = 10 * 1024 * 1024

def allowed_file(filename: str, selfie=False):
    ext = Path(filename).suffix.lower().lstrip(".")
    return ext in (SELFIE_EXT if selfie else DOCUMENT_EXT)

def secure_filename(filename):
    return secrets.token_hex(16) + Path(filename).suffix.lower()

async def save_upload_file_tmp(upload_file: UploadFile, tmp_dir: str):
    path = os.path.join(tmp_dir, secure_filename(upload_file.filename or "upload"))
    total = 0
    try:
        with open(path, "wb") as buffer:
            while chunk := await upload_file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_FILE_SIZE:
                    raise ValueError("File exceeds the 10 MB prototype limit.")
                buffer.write(chunk)
    finally:
        await upload_file.close()
    return path
