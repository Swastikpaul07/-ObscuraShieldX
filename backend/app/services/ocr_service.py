import re

import cv2
import numpy as np
from PIL import Image
from .paddle_ocr_service import run_paddle_ocr
from .sarvam_service import run_sarvam_digitise

try:
    import pytesseract
    from pytesseract import Output

    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False


def image_to_pil(img_cv):
    """Convert an OpenCV BGR image to a PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


def normalize_spaces(text):
    """Normalize repeated whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_aadhaar_number(text):
    """
    Convert common OCR variants of an Aadhaar-style number
    into a 12-digit string when possible.
    """
    digits = re.sub(r"\D", "", text)

    if len(digits) == 12:
        return digits

    return None


def extract_aadhaar_number(text):
    """
    Look for a 12-digit Aadhaar-style number.

    Handles:
        9876 5432 1098
        987654321098
        9876-5432-1098
    """
    patterns = [
        r"\b(\d{4})\s+(\d{4})\s+(\d{4})\b",
        r"\b(\d{4})-(\d{4})-(\d{4})\b",
        r"\b(\d{12})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            if len(match.groups()) == 3:
                number = "".join(match.groups())
            else:
                number = match.group(1)

            normalized = normalize_aadhaar_number(number)

            if normalized:
                return normalized

    return None


def extract_name(lines):
    """
    Extract a person's name from common ID-card layouts.

    Example:
        Cristiano Ronaldo
    """
    name_labels = [
        "name",
        "full name",
        "naam",
    ]

    for index, line in enumerate(lines):
        clean = line.strip()

        if not clean:
            continue

        lower = clean.lower()

        for label in name_labels:
            if lower.startswith(label):
                candidate = re.sub(
                    rf"^{re.escape(label)}\s*[:\-]?\s*",
                    "",
                    clean,
                    flags=re.IGNORECASE,
                ).strip()

                if is_reasonable_name(candidate):
                    return candidate

        # If the OCR line contains a known name label with text after it.
        if "name:" in lower:
            candidate = clean.split(":", 1)[1].strip()

            if is_reasonable_name(candidate):
                return candidate

    # Fallback:
    # Look for a line containing 2-4 alphabetic words.
    excluded_words = {
        "aadhaar",
        "aadhar",
        "male",
        "female",
        "india",
        "government",
        "address",
        "dob",
        "date",
        "birth",
        "year",
        "your",
    }

    for line in lines:
        candidate = line.strip()

        if not candidate:
            continue

        words = candidate.split()

        if 2 <= len(words) <= 4:
            if all(re.fullmatch(r"[A-Za-z.'-]+", word) for word in words):
                if not any(word.lower() in excluded_words for word in words):
                    return candidate

    return None


def is_reasonable_name(value):
    """Basic validation for a person-name candidate."""
    if not value:
        return False

    words = value.split()

    if not (1 <= len(words) <= 5):
        return False

    if len(value) < 3 or len(value) > 80:
        return False

    return all(
        re.fullmatch(r"[A-Za-z.'-]+", word)
        for word in words
    )


def extract_gender(text):
    """Extract gender from common ID-card wording."""
    lower = text.lower()

    if re.search(r"\bmale\b", lower):
        return "Male"

    if re.search(r"\bfemale\b", lower):
        return "Female"

    if re.search(r"\btransgender\b", lower):
        return "Transgender"

    return None


def extract_date_of_birth(text):
    """
    Extract common date formats.

    Examples:
        12/05/1999
        12-05-1999
        12.05.1999
        1999-05-12
    """
    patterns = [
        r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4}\b",
        r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2}\b",
        r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return match.group(0)

    return None


def extract_address(lines):
    for index, line in enumerate(lines):
        clean = line.strip()

        if not clean:
            continue

        lower = clean.lower()

        if lower.startswith("address"):
            candidate = re.sub(
                r"^address\s*[:\-]?\s*",
                "",
                clean,
                flags=re.IGNORECASE,
            ).strip()

            if candidate:
                return candidate

            if index + 1 < len(lines):
                next_line = lines[index + 1].strip()

                if next_line:
                    return next_line

    location_patterns = [
        r"\b[A-Za-z]+,\s*[A-Za-z]+,\s*India\b",
        r"\b[A-Za-z]+,\s*[A-Za-z]+\s+India\b",
        r"\b[A-Za-z]+,\s*India\b",
    ]

    for line in lines:
        clean = line.strip()

        if not clean:
            continue

        if clean.lower() == "government of india":
            continue

        for pattern in location_patterns:
            match = re.search(
                pattern,
                clean,
                re.IGNORECASE,
            )

            if match:
                return match.group(0)

    return None
  
def extract_fields_from_text(text):
    """
    Extract structured fields from OCR text.
    """
    normalized_text = normalize_spaces(text)

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    fields = {}

    name = extract_name(lines)
    if name:
        fields["name"] = name

    gender = extract_gender(normalized_text)
    if gender:
        fields["gender"] = gender

    dob = extract_date_of_birth(normalized_text)
    if dob:
        fields["date_of_birth"] = dob

    aadhaar_number = extract_aadhaar_number(text)
    if aadhaar_number:
        fields["document_number"] = aadhaar_number

    address = extract_address(lines)
    if address:
        fields["address"] = address

    return fields


def ocr_image(img_cv, source_path=None):
    """Run PaddleOCR first, with Tesseract as fallback."""

    # Try PaddleOCR first.
    if source_path:
        paddle_result = run_paddle_ocr(source_path)

        if paddle_result.get("success"):
            text = paddle_result.get("text", "")
            fields = extract_fields_from_text(text)

            return {
                "text": text,
                "fields": fields,
                "avg_confidence": paddle_result.get("average_confidence"),
                "error": None,
                "engine": "paddleocr",
                "lines": paddle_result.get("lines", []),
            }

    # Fall back to Tesseract if PaddleOCR fails.
    if not TESSERACT_AVAILABLE:
        return {
            "text": "",
            "fields": {},
            "avg_confidence": None,
            "error": "PaddleOCR failed and Tesseract is not installed.",
            "engine": "none",
            "lines": [],
        }

    try:
        pil_image = image_to_pil(img_cv)

        data = pytesseract.image_to_data(
            pil_image,
            output_type=Output.DICT,
        )

        confidences = []
        line_words = {}

        for i, (text_value, conf) in enumerate(
            zip(data.get("text", []), data.get("conf", []))
        ):
            text_value = text_value.strip()

            if not text_value:
                continue

            try:
                confidence = float(conf)

                if confidence >= 0:
                    confidences.append(
                        max(0.0, min(100.0, confidence)) / 100.0
                    )
            except (ValueError, TypeError):
                pass

            block_num = data.get("block_num", [0])[i]
            par_num = data.get("par_num", [0])[i]
            line_num = data.get("line_num", [0])[i]

            line_key = (block_num, par_num, line_num)

            line_words.setdefault(line_key, [])
            line_words[line_key].append(text_value)

        lines = [
            " ".join(words)
            for words in line_words.values()
            if words
        ]

        full_text = "\n".join(lines)
        fields = extract_fields_from_text(full_text)

        return {
            "text": full_text,
            "fields": fields,
            "avg_confidence": (
                float(np.mean(confidences))
                if confidences
                else None
            ),
            "error": None,
            "engine": "tesseract",
            "lines": lines,
        }

    except Exception as exc:
        return {
            "text": "",
            "fields": {},
            "avg_confidence": None,
            "error": str(exc),
            "engine": "none",
            "lines": [],
        }
def hybrid_ocr(img_cv, source_path=None):
    local_result = ocr_image(
        img_cv,
        source_path=source_path,
    )

    sarvam_result = {
        "success": False,
        "text": "",
        "blocks": [],
        "pages": [],
        "job_id": None,
        "error": "Source path not provided.",
    }

    if source_path:
        sarvam_result = run_sarvam_digitise(
            source_path
        )

    if sarvam_result.get("success"):
        sarvam_text = sarvam_result.get("text", "")

        sarvam_fields = extract_fields_from_text(
            sarvam_text
        )

        combined_fields = dict(
            local_result.get("fields", {})
        )

        for key, value in sarvam_fields.items():
            if value:
                combined_fields[key] = value

        return {
            "text": sarvam_text,
            "fields": combined_fields,
            "avg_confidence": local_result.get(
                "avg_confidence"
            ),
            "error": None,
            "engine": "hybrid",
            "local_engine": local_result.get(
                "engine"
            ),
            "local_text": local_result.get(
                "text", ""
            ),
            "sarvam_text": sarvam_text,
            "sarvam_blocks": sarvam_result.get(
                "blocks", []
            ),
            "sarvam_job_id": sarvam_result.get(
                "job_id"
            ),
        }

    return {
        **local_result,
        "engine": "hybrid_local_fallback",
        "sarvam_text": "",
        "sarvam_blocks": [],
        "sarvam_job_id": sarvam_result.get(
            "job_id"
        ),
        "sarvam_error": sarvam_result.get(
            "error"
        ),
    }