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
from core.audit import setup_logging

setup_logging()
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
    """Extract the authenticated session from the request cookie.

    Looks up the session ID stored in the HttpOnly ``crb_session`` cookie and
    returns the in-database session dict, or ``None`` when no valid session
    exists (unauthenticated request).

    Args:
        request: The incoming FastAPI :class:`~fastapi.Request`.

    Returns:
        A dict with ``'login'``, ``'avatar_url'``, and ``'access_token'`` keys,
        or ``None`` if the request is unauthenticated.
    """
    return github_auth.get_session(request.cookies.get(SESSION_COOKIE))


@app.on_event("startup")
def _startup() -> None:
    """FastAPI lifecycle hook — initialise the database on first startup.

    Creates all SQLAlchemy tables and runs any pending lightweight migrations.
    Called once automatically by FastAPI before the first request is served.
    """
    init_db()
    logger.info("Application started — database initialised")


def _scan_to_dict(scan: Scan, include_findings: bool = False) -> dict:
    """Serialise a :class:`~core.database.Scan` ORM row to a plain dict.

    Args:
        scan:             The ORM row to serialise.
        include_findings: When ``True``, also fetch and embed the associated
                          :class:`~core.database.ScanFindingRow` rows and the
                          file list under ``'findings'`` and ``'files'`` keys.

    Returns:
        A JSON-serialisable dict representing the scan’s current state.
    """
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
                logger.warning("Could not parse file_list JSON for scan_id=%s", scan.id)
                data["files"] = []
        return data
    finally:
        db.close()


async def _run_and_track(repo_source: str, scan_id: str) -> None:
    """Background task wrapper that runs a full scan and catches unhandled errors.

    Wraps :func:`~agents.repo_orchestrator.run_scan` so that any exception that
    escapes the orchestrator is logged before it propagates, preventing the
    asyncio task from silently dying.

    Args:
        repo_source: GitHub repository URL to scan.
        scan_id:     Pre-assigned scan ID for the new scan record.
    """
    try:
        await run_scan(repo_source, scan_id=scan_id)
    except Exception:
        logger.error(
            "Background scan raised an unhandled exception — scan_id=%s repo=%s",
            scan_id, repo_source, exc_info=True,
        )


@app.post("/api/scan")
async def create_scan(req: ScanRequest) -> dict:
    if not req.repo_source.strip():
        logger.warning("POST /api/scan rejected — empty repo_source")
        raise HTTPException(400, "repo_source is required")
    if not _is_git_url(req.repo_source.strip()):
        logger.warning("POST /api/scan rejected — not a Git URL: %s", req.repo_source)
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

    logger.info("Scan queued — scan_id=%s repo=%s", scan_id, req.repo_source)
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
            logger.warning("GET /api/scan/%s — scan not found", scan_id)
            raise HTTPException(404, "scan not found")
        return _scan_to_dict(scan, include_findings=True)
    finally:
        db.close()


@app.post("/api/scan/{scan_id}/fix")
async def create_fix(scan_id: str, req: FixRequest, request: Request) -> dict:
    """Generate a concrete, committable code change for one finding."""
    session = _current_session(request)
    if not session:
        logger.warning(
            "Unauthenticated fix request — scan_id=%s finding_id=%s",
            scan_id, req.finding_id,
        )
    token = session["access_token"] if session else None
    try:
        logger.info("Fix requested — scan_id=%s finding_id=%s", scan_id, req.finding_id)
        return await generate_fix(scan_id, req.finding_id, token)
    except ValueError as e:
        logger.error(
            "Fix generation failed — scan_id=%s finding_id=%s: %s",
            scan_id, req.finding_id, e,
        )
        raise HTTPException(400, str(e))


@app.post("/api/scan/{scan_id}/commit")
async def commit_fixes(scan_id: str, req: CommitRequest, request: Request) -> dict:
    """Apply the selected fixes and commit them together (atomic)."""
    if not req.finding_ids:
        raise HTTPException(400, "finding_ids is required")
    session = _current_session(request)
    if not session:
        logger.warning(
            "Unauthenticated commit attempt — scan_id=%s findings=%s",
            scan_id, req.finding_ids,
        )
    token = session["access_token"] if session else None
    login = session["login"] if session else None
    try:
        logger.info(
            "Commit requested — scan_id=%s user=%s fixes=%d mode=%s",
            scan_id, login or "anonymous", len(req.finding_ids), req.mode,
        )
        result = apply_and_commit(scan_id, req.finding_ids, req.message,
                                  mode=req.mode, token=token, login=login)
        if result.get("committed"):
            logger.info(
                "Commit applied — scan_id=%s user=%s sha=%s mode=%s",
                scan_id, login, result.get("short_sha"), req.mode,
            )
        else:
            logger.warning(
                "Commit rejected — scan_id=%s user=%s reason=%s",
                scan_id, login, result.get("message"),
            )
        return result
    except ValueError as e:
        logger.error("Commit failed — scan_id=%s user=%s: %s", scan_id, login, e)
        raise HTTPException(400, str(e))


@app.get("/api/scan/{scan_id}/access")
def scan_access(scan_id: str, request: Request) -> dict:
    """Tell the UI whether the current user can commit fixes for this scan.

    Checks GitHub push permissions for the scanned repository and returns a
    structured response the frontend uses to enable or disable the commit
    button.  Unauthenticated users receive ``login_required: true``.
    """
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
    """Return the currently authenticated user, or an unauthenticated indicator.

    Used by the React frontend on load to decide whether to show the login
    button or the user avatar.  Always returns a 200 — the ``authenticated``
    field conveys the result.
    """
    session = _current_session(request)
    if not session:
        return {"authenticated": False, "oauth_enabled": GITHUB_OAUTH_ENABLED}
    return {"authenticated": True, "login": session["login"],
            "avatar_url": session["avatar_url"], "oauth_enabled": True}


@app.get("/api/auth/github/login")
def auth_login():
    if not GITHUB_OAUTH_ENABLED:
        logger.warning("GitHub OAuth login attempted but OAuth is not configured")
        raise HTTPException(400, "GitHub OAuth is not configured on the server")
    logger.info("GitHub OAuth login flow initiated")
    return RedirectResponse(github_auth.build_authorize_url())


@app.get("/api/auth/github/callback")
def auth_callback(code: str = "", state: str = ""):
    """Handle the GitHub OAuth callback (step 2 of the OAuth flow).

    Validates the CSRF ``state`` token, exchanges the one-time ``code`` for an
    access token, fetches the authenticated user’s profile, persists a
    server-side session, and sets an HttpOnly session cookie.  On any failure
    the browser is redirected to ``/?auth=error`` so the frontend can surface
    a user-facing message.
    """
    """Handle the GitHub OAuth callback (step 2 of the OAuth flow).

    Validates the CSRF ``state`` token, exchanges the one-time ``code`` for an
    access token, fetches the authenticated user’s profile, persists a
    server-side session, and sets an HttpOnly session cookie.  On any failure
    the browser is redirected to ``/?auth=error`` so the frontend can surface
    a user-facing message.
    """
    if not GITHUB_OAUTH_ENABLED:
        raise HTTPException(400, "GitHub OAuth is not configured on the server")
    if not code or not github_auth.consume_state(state):
        logger.warning("OAuth callback received invalid or expired code/state")
        return RedirectResponse(f"{FRONTEND_URL}/?auth=error")
    token = github_auth.exchange_code_for_token(code)
    if not token:
        logger.error("OAuth token exchange failed — no token returned by GitHub")
        return RedirectResponse(f"{FRONTEND_URL}/?auth=error")
    user = github_auth.get_authenticated_user(token)
    if not user:
        logger.error("OAuth authentication failed — could not retrieve user info after token exchange")
        return RedirectResponse(f"{FRONTEND_URL}/?auth=error")
    sid = github_auth.create_session(user["login"], user["avatar_url"], token)
    logger.info("User logged in via GitHub OAuth — login=%s", user["login"])
    resp = RedirectResponse(f"{FRONTEND_URL}/?auth=ok")
    resp.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax",
                    max_age=60 * 60 * 24 * 7, path="/")
    return resp


@app.post("/api/auth/logout")
def auth_logout(request: Request, response: Response) -> dict:
    session = _current_session(request)
    login = session["login"] if session else "anonymous"
    logger.info("User logged out — login=%s", login)
    github_auth.delete_session(request.cookies.get(SESSION_COOKIE))
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"authenticated": False}


@app.get("/api/scan/{scan_id}/report.html", response_class=HTMLResponse)
def get_report_html(scan_id: str) -> str:
    db = SessionLocal()
    try:
        scan = db.query(Scan).filter(Scan.id == scan_id).first()
        if not scan or not scan.report_html:
            logger.warning("HTML report not ready — scan_id=%s", scan_id)
            raise HTTPException(404, "report not ready")
        return scan.report_html
    finally:
        db.close()


@app.get("/api/scans")
def list_scans() -> list[dict]:
    """Return the 25 most recent scans ordered newest-first.

    Used by the React dashboard to populate the scan history list on load.
    Does not include per-scan findings to keep the payload small.
    """
    """Return the 25 most recent scans ordered newest-first.

    Used by the React dashboard to populate the scan history list on load.
    Does not include per-scan findings to keep the payload small.
    """
    db = SessionLocal()
    try:
        scans = db.query(Scan).order_by(Scan.created_at.desc()).limit(25).all()
        return [_scan_to_dict(s) for s in scans]
    finally:
        db.close()
