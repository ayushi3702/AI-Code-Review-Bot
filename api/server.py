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
import json
import logging

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from core.config import (
    GITHUB_OAUTH_ENABLED, FRONTEND_URL, SESSION_COOKIE,
)
from core.database import SessionLocal, Scan, ScanFindingRow, init_db
from core.fixer import generate_fix, apply_and_commit, _is_git_url
from core import github_auth
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


class FixRequest(BaseModel):
    finding_id: str


class CommitRequest(BaseModel):
    finding_ids: list[str]
    message: str = ""
    mode: str = "pr"   # direct | pr


def _current_session(request: Request) -> dict | None:
    return github_auth.get_session(request.cookies.get(SESSION_COOKIE))


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
            "committable": False,  # resolved per-user via /access (GitHub push rights)
        }
        if include_findings:
            rows = db.query(ScanFindingRow).filter(ScanFindingRow.scan_id == scan.id).all()
            data["findings"] = [{
                "id": r.id,
                "agent": r.agent, "severity": r.severity, "file": r.file,
                "line": r.line, "title": r.title, "detail": r.detail,
                "recommendation": r.recommendation,
            } for r in rows]
            try:
                data["files"] = json.loads(scan.file_list) if scan.file_list else []
            except (json.JSONDecodeError, TypeError):
                data["files"] = []
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
    if not _is_git_url(req.repo_source.strip()):
        raise HTTPException(
            400,
            "Only GitHub repository URLs are supported "
            "(e.g. https://github.com/owner/repo.git).",
        )

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


@app.post("/api/scan/{scan_id}/fix")
async def create_fix(scan_id: str, req: FixRequest, request: Request) -> dict:
    """Generate a concrete, committable code change for one finding."""
    session = _current_session(request)
    token = session["access_token"] if session else None
    try:
        return await generate_fix(scan_id, req.finding_id, token)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/scan/{scan_id}/commit")
async def commit_fixes(scan_id: str, req: CommitRequest, request: Request) -> dict:
    """Apply the selected fixes and commit them together (atomic)."""
    if not req.finding_ids:
        raise HTTPException(400, "finding_ids is required")
    session = _current_session(request)
    token = session["access_token"] if session else None
    login = session["login"] if session else None
    try:
        return apply_and_commit(scan_id, req.finding_ids, req.message,
                                mode=req.mode, token=token, login=login)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/scan/{scan_id}/access")
def scan_access(scan_id: str, request: Request) -> dict:
    """Tell the UI whether (and how) the current user can commit this scan."""
    db = SessionLocal()
    try:
        scan = db.query(Scan).filter(Scan.id == scan_id).first()
        if not scan:
            raise HTTPException(404, "scan not found")
        source = scan.repo_source
    finally:
        db.close()

    parsed = github_auth.parse_github_repo(source)
    owner, repo = parsed if parsed else (None, None)
    session = _current_session(request)
    if not session:
        return {"mode": "github", "committable": False, "login_required": True,
                "oauth_enabled": GITHUB_OAUTH_ENABLED, "owner": owner, "repo": repo}

    if not parsed:
        return {"mode": "github", "committable": False, "login_required": False,
                "reason": "not a recognized GitHub repository"}
    push_ok, default_branch, reason = github_auth.can_push(session["access_token"], owner, repo)
    return {"mode": "github", "committable": push_ok, "can_push": push_ok,
            "login_required": False, "owner": owner, "repo": repo,
            "default_branch": default_branch, "reason": reason,
            "login": session["login"]}


# ── GitHub OAuth ─────────────────────────────────────────────────────────────

@app.get("/api/auth/me")
def auth_me(request: Request) -> dict:
    session = _current_session(request)
    if not session:
        return {"authenticated": False, "oauth_enabled": GITHUB_OAUTH_ENABLED}
    return {"authenticated": True, "login": session["login"],
            "avatar_url": session["avatar_url"], "oauth_enabled": True}


@app.get("/api/auth/github/login")
def auth_login():
    if not GITHUB_OAUTH_ENABLED:
        raise HTTPException(400, "GitHub OAuth is not configured on the server")
    return RedirectResponse(github_auth.build_authorize_url())


@app.get("/api/auth/github/callback")
def auth_callback(code: str = "", state: str = ""):
    if not GITHUB_OAUTH_ENABLED:
        raise HTTPException(400, "GitHub OAuth is not configured on the server")
    if not code or not github_auth.consume_state(state):
        return RedirectResponse(f"{FRONTEND_URL}/?auth=error")
    token = github_auth.exchange_code_for_token(code)
    if not token:
        return RedirectResponse(f"{FRONTEND_URL}/?auth=error")
    user = github_auth.get_authenticated_user(token)
    if not user:
        return RedirectResponse(f"{FRONTEND_URL}/?auth=error")
    sid = github_auth.create_session(user["login"], user["avatar_url"], token)
    resp = RedirectResponse(f"{FRONTEND_URL}/?auth=ok")
    resp.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax",
                    max_age=60 * 60 * 24 * 7, path="/")
    return resp


@app.post("/api/auth/logout")
def auth_logout(request: Request, response: Response) -> dict:
    github_auth.delete_session(request.cookies.get(SESSION_COOKIE))
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"authenticated": False}


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
