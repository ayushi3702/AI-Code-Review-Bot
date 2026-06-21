"""FastAPI server for the repo-wide deep-scan platform.

Endpoints (consumed by the React frontend):
    POST /api/scan            { "repo_source": "<url|path>" } → { scan_id }
    GET  /api/scan/{id}       → status, stage, score, grade, findings
    GET  /api/scan/{id}/report.html  → standalone HTML report
    GET  /api/scans           → recent scans

Scans run as background asyncio tasks so the POST returns immediately and the
UI polls GET /api/scan/{id} for progress. The same run_scan() entry point is
shared with the CLI.
"""
from __future__ import annotations
import asyncio
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from core.database import SessionLocal, Scan, ScanFindingRow, init_db
from agents.repo_orchestrator import run_scan

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Code Review Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# keep references so background tasks aren't garbage-collected mid-run
_TASKS: set[asyncio.Task] = set()


class ScanRequest(BaseModel):
    repo_source: str


@app.on_event("startup")
def _startup() -> None:
    init_db()


def _scan_to_dict(scan: Scan, include_findings: bool = False) -> dict:
    db = SessionLocal()
    try:
        data = {
            "scan_id": scan.id,
            "repo_source": scan.repo_source,
            "repo_name": scan.repo_name,
            "status": scan.status,
            "stage": scan.stage,
            "file_count": scan.file_count,
            "chunk_count": scan.chunk_count,
            "finding_count": scan.finding_count,
            "score": scan.score,
            "grade": scan.grade,
            "error": scan.error,
            "created_at": scan.created_at.isoformat() if scan.created_at else None,
        }
        if include_findings:
            rows = db.query(ScanFindingRow).filter(ScanFindingRow.scan_id == scan.id).all()
            data["findings"] = [{
                "agent": r.agent, "severity": r.severity, "file": r.file,
                "line": r.line, "title": r.title, "detail": r.detail,
                "recommendation": r.recommendation,
            } for r in rows]
        return data
    finally:
        db.close()


async def _run_and_track(repo_source: str, scan_id: str) -> None:
    try:
        await run_scan(repo_source, scan_id=scan_id)
    except Exception:
        logger.exception("Background scan %s failed", scan_id)


@app.post("/api/scan")
async def create_scan(req: ScanRequest) -> dict:
    if not req.repo_source.strip():
        raise HTTPException(400, "repo_source is required")

    import uuid
    scan_id = str(uuid.uuid4())
    db = SessionLocal()
    try:
        db.add(Scan(id=scan_id, repo_source=req.repo_source, status="queued", stage="queued"))
        db.commit()
    finally:
        db.close()

    task = asyncio.create_task(_run_and_track(req.repo_source, scan_id))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)

    return {"scan_id": scan_id, "status": "queued"}


@app.get("/api/scan/{scan_id}")
def get_scan(scan_id: str) -> dict:
    db = SessionLocal()
    try:
        scan = db.query(Scan).filter(Scan.id == scan_id).first()
        if not scan:
            raise HTTPException(404, "scan not found")
        return _scan_to_dict(scan, include_findings=True)
    finally:
        db.close()


@app.get("/api/scan/{scan_id}/report.html", response_class=HTMLResponse)
def get_report_html(scan_id: str) -> str:
    db = SessionLocal()
    try:
        scan = db.query(Scan).filter(Scan.id == scan_id).first()
        if not scan or not scan.report_html:
            raise HTTPException(404, "report not ready")
        return scan.report_html
    finally:
        db.close()


@app.get("/api/scans")
def list_scans() -> list[dict]:
    db = SessionLocal()
    try:
        scans = db.query(Scan).order_by(Scan.created_at.desc()).limit(25).all()
        return [_scan_to_dict(s) for s in scans]
    finally:
        db.close()
