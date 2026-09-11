# ShieldX — Full-Stack Hackathon Prototype

ShieldX is an AI-assisted document screening and identity verification prototype. It demonstrates a complete flow from document upload to OCR, image anomaly heuristics, optional face comparison, explainable risk scoring, database persistence, and a browser dashboard.

> **Important:** This is a hackathon prototype. It is not legally authoritative identity verification and should not be used for real border/security decisions without substantial validation, security review, privacy controls, and qualified oversight.

## Stack

- Python 3.11
- FastAPI + Uvicorn
- SQLAlchemy
- SQLite locally / PostgreSQL-compatible via `DATABASE_URL`
- OpenCV + Pillow
- Tesseract OCR
- pdf2image + Poppler
- React is not required for this MVP: the frontend is a lightweight HTML/CSS/JS dashboard served directly by FastAPI.
- Docker / Render-ready configuration

## Run locally on Windows

From the ShieldX root:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r backend\requirements.txt
```

Install Tesseract OCR and add it to PATH if you want OCR locally. For PDFs, install Poppler and add it to PATH.

Then:

```powershell
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

- Dashboard: http://127.0.0.1:8000
- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## GitHub

```powershell
git init -b main
git add .
git commit -m "Build ShieldX full-stack prototype"
git remote add origin https://github.com/YOUR_USERNAME/ShieldX.git
git push -u origin main
```

Never commit `.env`, passwords, API keys, or identity documents.

## Docker

```powershell
docker compose up --build
```

Then open http://localhost:8000.

## API

`POST /api/v1/verify/document`

Multipart fields:

- `document`: JPG, JPEG, PNG, PDF
- `selfie`: optional JPG, JPEG, PNG

Additional endpoints:

- `GET /health`
- `GET /api/v1/screenings`
- `GET /api/v1/dashboard/stats`

## Architecture

```text
Browser
   |
   | multipart upload
   v
FastAPI
   |
   +--> Preprocessing
   +--> OCR / Tesseract
   +--> Anomaly heuristics
   +--> Optional face comparison
   +--> Explainable risk engine
   |
   v
SQLAlchemy
   |
   v
SQLite (local) / PostgreSQL (deployment)
```

## What the prototype does NOT claim

The current face comparison uses OpenCV Haar face detection + SSIM and is not a production biometric verification method. The anomaly detector is heuristic-based, and the risk score is not statistically validated. These are intentionally replaceable prototype components.

## Recommended hackathon demo

1. Open the ShieldX dashboard.
2. Show the live system status and dashboard counters.
3. Upload a sample identity document.
4. Optionally add a selfie.
5. Click **Analyze Document**.
6. Explain the OCR, anomaly, face, and risk signals.
7. Show the explainable reasons and screening history.
8. Open `/docs` to demonstrate the FastAPI API.
9. Explain that the repository is version-controlled with Git and published to GitHub.
