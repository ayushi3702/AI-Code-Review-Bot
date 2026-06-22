"""GitHub OAuth + REST helpers and server-side session storage.

Uses only the standard library (urllib) so it adds no new dependencies. The
OAuth token is kept server-side in the `github_sessions` table and referenced by
a random, HttpOnly cookie value — it is never exposed to the browser.
"""
from __future__ import annotations
import re
import json
import time
import secrets
import logging
import datetime
import urllib.parse
import urllib.request
import urllib.error

from core.config import (
    GITHUB_CLIENT_ID,
    GITHUB_CLIENT_SECRET,
    GITHUB_OAUTH_CALLBACK,
    GITHUB_OAUTH_SCOPE,
)
from core.database import SessionLocal, GitHubSession

logger = logging.getLogger(__name__)

_AUTHORIZE = "https://github.com/login/oauth/authorize"
_TOKEN = "https://github.com/login/oauth/access_token"
_API = "https://api.github.com"

# short-lived CSRF states for the OAuth round-trip: {state: created_at}
_STATES: dict[str, float] = {}
_STATE_TTL = 600  # seconds


# ── repo URL parsing ─────────────────────────────────────────────────────────

def parse_github_repo(url: str) -> tuple[str, str] | None:
    """Return (owner, repo) for a GitHub URL, or None if it isn't one."""
    if not url:
        return None
    m = re.search(r"github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?/?$", url.strip())
    if not m:
        return None
    return m.group(1), m.group(2)


# ── low-level HTTP (stdlib) ──────────────────────────────────────────────────

def _request(method: str, url: str, *, token: str | None = None,
             data: dict | None = None, accept: str = "application/vnd.github+json"):
    headers = {"Accept": accept, "User-Agent": "AI-Code-Review-Bot"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = resp.read().decode()
            return resp.status, (json.loads(payload) if payload else {})
    except urllib.error.HTTPError as e:
        payload = e.read().decode() if e.fp else ""
        try:
            parsed = json.loads(payload) if payload else {}
        except json.JSONDecodeError:
            parsed = {"message": payload}
        return e.code, parsed
    except urllib.error.URLError as e:
        return 0, {"message": str(e)}


# ── OAuth flow ───────────────────────────────────────────────────────────────

def build_authorize_url() -> str:
    state = secrets.token_urlsafe(24)
    _STATES[state] = time.time()
    # opportunistic cleanup of expired states
    for s, ts in list(_STATES.items()):
        if time.time() - ts > _STATE_TTL:
            _STATES.pop(s, None)
    params = urllib.parse.urlencode({
        "client_id": GITHUB_CLIENT_ID,
        "redirect_uri": GITHUB_OAUTH_CALLBACK,
        "scope": GITHUB_OAUTH_SCOPE,
        "state": state,
    })
    return f"{_AUTHORIZE}?{params}"


def consume_state(state: str) -> bool:
    ts = _STATES.pop(state, None)
    return ts is not None and (time.time() - ts) <= _STATE_TTL


def exchange_code_for_token(code: str) -> str | None:
    data = {
        "client_id": GITHUB_CLIENT_ID,
        "client_secret": GITHUB_CLIENT_SECRET,
        "code": code,
        "redirect_uri": GITHUB_OAUTH_CALLBACK,
    }
    status, payload = _request("POST", _TOKEN, data=data)
    if status == 200 and payload.get("access_token"):
        return payload["access_token"]
    logger.warning("OAuth token exchange failed: %s %s", status, payload)
    return None


# ── GitHub REST helpers ──────────────────────────────────────────────────────

def get_authenticated_user(token: str) -> dict | None:
    status, payload = _request("GET", f"{_API}/user", token=token)
    if status == 200:
        return {"login": payload.get("login"), "avatar_url": payload.get("avatar_url")}
    return None


def get_repo_info(token: str, owner: str, repo: str) -> tuple[int, dict]:
    return _request("GET", f"{_API}/repos/{owner}/{repo}", token=token)


def can_push(token: str, owner: str, repo: str) -> tuple[bool, str | None, str | None]:
    """Return (can_push, default_branch, reason)."""
    status, payload = get_repo_info(token, owner, repo)
    if status == 200:
        perms = payload.get("permissions", {})
        if perms.get("push") or perms.get("admin") or perms.get("maintain"):
            return True, payload.get("default_branch", "main"), None
        return False, payload.get("default_branch", "main"), "you don't have push access to this repository"
    if status == 404:
        return False, None, "repository not found or you lack access"
    return False, None, payload.get("message", f"GitHub API error {status}")


def create_pull_request(token: str, owner: str, repo: str, head: str,
                        base: str, title: str, body: str) -> tuple[int, dict]:
    return _request("POST", f"{_API}/repos/{owner}/{repo}/pulls", token=token,
                    data={"title": title, "head": head, "base": base, "body": body})


# ── server-side sessions ─────────────────────────────────────────────────────

def create_session(login: str, avatar_url: str, access_token: str) -> str:
    sid = secrets.token_urlsafe(32)
    db = SessionLocal()
    try:
        db.add(GitHubSession(id=sid, login=login, avatar_url=avatar_url,
                             access_token=access_token,
                             created_at=datetime.datetime.utcnow()))
        db.commit()
    finally:
        db.close()
    return sid


def get_session(sid: str | None) -> dict | None:
    if not sid:
        return None
    db = SessionLocal()
    try:
        row = db.query(GitHubSession).filter(GitHubSession.id == sid).first()
        if not row:
            return None
        return {"login": row.login, "avatar_url": row.avatar_url,
                "access_token": row.access_token}
    finally:
        db.close()


def delete_session(sid: str | None) -> None:
    if not sid:
        return
    db = SessionLocal()
    try:
        db.query(GitHubSession).filter(GitHubSession.id == sid).delete()
        db.commit()
    finally:
        db.close()
