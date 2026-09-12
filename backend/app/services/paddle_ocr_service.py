import os
from pathlib import Path

# Workaround for PaddlePaddle 3.x Windows CPU/oneDNN
# compatibility issue observed during local inference.
os.environ["FLAGS_enable_pir_api"] = "0"

from paddleocr import PaddleOCR


# Create the PaddleOCR engine once.
ocr_engine = PaddleOCR(
    lang="en",
    device="cpu",
    enable_mkldnn=False,
)


def run_paddle_ocr(image_path: str) -> dict:
    """
    Run PaddleOCR and return structured OCR information.

    Returns:
        success
        text
        lines
        average_confidence
        detected_text_regions
        error
    """

    path = Path(image_path)

    if not path.exists():
        return {
            "success": False,
            "text": "",
            "lines": [],
            "average_confidence": None,
            "detected_text_regions": 0,
            "error": f"File not found: {image_path}",
        }

    try:
        result = ocr_engine.predict(str(path))

        lines = []
        all_text = []
        confidence_values = []

        for page in result:
            data = page.json

            if not data:
                continue

            res = data.get("res", {})

            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            boxes = res.get("rec_boxes", [])

            for index, text in enumerate(texts):
                text = str(text).strip()

                if not text:
                    continue

                confidence = None

                if index < len(scores):
                    try:
                        confidence = float(scores[index])
                        confidence = max(
                            0.0,
                            min(1.0, confidence),
                        )

                        confidence_values.append(confidence)

                    except (ValueError, TypeError):
                        confidence = None

                box = None

                if index < len(boxes):
                    try:
                        box = [
                            int(value)
                            for value in boxes[index]
                        ]
                    except (ValueError, TypeError):
                        box = None

                lines.append(
                    {
                        "text": text,
                        "confidence": confidence,
                        "box": box,
                    }
                )

                all_text.append(text)

        average_confidence = None

        if confidence_values:
            average_confidence = (
                sum(confidence_values)
                / len(confidence_values)
            )

        return {
            "success": True,
            "text": "\n".join(all_text),
            "lines": lines,
            "average_confidence": average_confidence,
            "detected_text_regions": len(lines),
            "error": None,
        }

    except Exception as exc:
        return {
            "success": False,
            "text": "",
            "lines": [],
            "average_confidence": None,
            "detected_text_regions": 0,
            "error": str(exc),
        }