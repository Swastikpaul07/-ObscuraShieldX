import os
import time
from pathlib import Path

from dotenv import load_dotenv
from sarvamai import SarvamAI


load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

if not SARVAM_API_KEY:
    raise RuntimeError("SARVAM_API_KEY is not configured.")


client = SarvamAI(
    api_subscription_key=SARVAM_API_KEY
)


def _get_mime_type(file_path: str) -> str:
    """
    Return the MIME type required by Sarvam Document AI.
    """

    suffix = Path(file_path).suffix.lower()

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".pdf": "application/pdf",
    }

    return mime_types.get(suffix, "application/octet-stream")


def run_sarvam_digitise(
    file_path: str,
    language: str = "en-IN",
    output_format: str = "md",
    timeout_seconds: int = 60,
    poll_interval: float = 2.0,
) -> dict:
    """
    Send a document to Sarvam Document AI, wait for completion,
    and return normalized OCR/layout information.
    """

    path = Path(file_path)

    if not path.exists():
        return {
            "success": False,
            "text": "",
            "blocks": [],
            "pages": [],
            "job_id": None,
            "error": f"File not found: {file_path}",
        }

    mime_type = _get_mime_type(file_path)

    try:
        with path.open("rb") as file:
            job = client.doc_ai.digitise(
                file=[
                    (
                        path.name,
                        file,
                        mime_type,
                    )
                ],
                language=language,
                output_format=output_format,
            )

        job_id = job.job_id

        start_time = time.time()

        while True:
            status = client.doc_ai.get_status(
                job_id=job_id
            )

            if status.status == "completed":
                break

            if status.status in {
                "failed",
                "cancelled",
                "canceled",
            }:
                return {
                    "success": False,
                    "text": "",
                    "blocks": [],
                    "pages": [],
                    "job_id": job_id,
                    "error": f"Sarvam job ended with status: {status.status}",
                }

            if time.time() - start_time >= timeout_seconds:
                return {
                    "success": False,
                    "text": "",
                    "blocks": [],
                    "pages": [],
                    "job_id": job_id,
                    "error": "Sarvam Document AI timed out.",
                }

            time.sleep(poll_interval)

        result = client.doc_ai.get_results(
            job_id=job_id
        )

        all_blocks = []
        all_pages = []
        all_text = []

        for document in result.documents or []:
            for page in document.pages or []:

                page_blocks = page.blocks or []

                page_text_parts = []

                for block in page_blocks:
                    text = str(
                        block.get("text", "")
                    ).strip()

                    if not text:
                        continue

                    normalized_block = {
                        "block_id": block.get("block_id"),
                        "text": text,
                        "layout_tag": block.get("layout_tag"),
                        "reading_order": block.get(
                            "reading_order"
                        ),
                        "coordinates": block.get(
                            "coordinates"
                        ),
                        "bbox_norm": block.get(
                            "bbox_norm"
                        ),
                    }

                    all_blocks.append(
                        normalized_block
                    )

                    page_text_parts.append(text)

                page_text = "\n".join(
                    page_text_parts
                )

                if page_text:
                    all_text.append(page_text)

                all_pages.append(
                    {
                        "page_number": getattr(
                            page,
                            "page_num",
                            None,
                        ),
                        "image_width": getattr(
                            page,
                            "image_width",
                            None,
                        ),
                        "image_height": getattr(
                            page,
                            "image_height",
                            None,
                        ),
                        "text": page_text,
                        "blocks": page_blocks,
                    }
                )

        return {
            "success": True,
            "text": "\n".join(all_text),
            "blocks": all_blocks,
            "pages": all_pages,
            "job_id": job_id,
            "error": None,
        }

    except Exception as exc:
        return {
            "success": False,
            "text": "",
            "blocks": [],
            "pages": [],
            "job_id": None,
            "error": str(exc),
        }