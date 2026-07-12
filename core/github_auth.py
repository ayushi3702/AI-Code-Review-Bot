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
    """Extract the ``(owner, repo)`` pair from a GitHub URL.

    Handles HTTPS clone URLs (``https://github.com/owner/repo.git``), SSH URLs
    (``git@github.com:owner/repo.git``), and plain browser URLs.

    Args:
        url: Any GitHub repository URL or empty string.

    Returns:
        A ``(owner, repo)`` tuple, or ``None`` if the URL is not a recognized
        GitHub URL.
    """
    if not url:
        return None
    m = re.search(r"github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?/?$", url.strip())
    if not m:
        return None
    return m.group(1), m.group(2)


# ── low-level HTTP (stdlib) ──────────────────────────────────────────────────

def _request(method: str, url: str, *, token: str | None = None,
             data: dict | None = None, accept: str = "application/vnd.github+json"):
    """Execute a GitHub API request using only the Python standard library.

    Avoids adding an ``httpx`` or ``requests`` dependency by using
    :mod:`urllib.request` directly.  JSON bodies are serialised automatically
    when ``data`` is provided.

    Args:
        method: HTTP verb — ``'GET'``, ``'POST'``, etc.
        url:    Full request URL.
        token:  Optional Bearer token added as an ``Authorization`` header.
        data:   Optional dict serialised to JSON and sent as the request body.
        accept: Value for the ``Accept`` header (defaults to GitHub’s v3 type).

    Returns:
        A ``(status_code, payload_dict)`` tuple.  On network failure the
        status code is ``0``; on HTTP errors it is the HTTP status code.  In
        both error cases ``payload`` contains at least a ``"message"`` key.
    """
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
        logger.warning("GitHub API HTTP %d at %s: %s", e.code, url, parsed.get("message", ""))
        return e.code, parsed
    except urllib.error.URLError as e:
        logger.error("GitHub API network error at %s: %s", url, e)
        return 0, {"message": str(e)}


# ── OAuth flow ───────────────────────────────────────────────────────────────

def build_authorize_url() -> str:
    """Build the GitHub OAuth authorisation redirect URL (step 1 of the OAuth flow).

    Generates a cryptographically random CSRF ``state`` token, stores it in an
    in-process dict with its creation timestamp, and embeds it in the URL.
    Expired states (older than ``_STATE_TTL`` seconds) are pruned
    opportunistically on each call to prevent unbounded memory growth.

    Returns:
        The full ``https://github.com/login/oauth/authorize?...`` URL to
        redirect the browser to.
    """
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
    """Validate and consume a one-time CSRF ``state`` token from an OAuth callback.

    Removes the token from the in-process store so it cannot be replayed, then
    checks that it was issued within ``_STATE_TTL`` seconds.

    Args:
        state: The ``state`` query parameter received in the OAuth callback URL.

    Returns:
        ``True`` if the token was present and unexpired; ``False`` otherwise.
    """
    ts = _STATES.pop(state, None)
    return ts is not None and (time.time() - ts) <= _STATE_TTL


def exchange_code_for_token(code: str) -> str | None:
    """Exchange a one-time OAuth authorisation code for a GitHub access token.

    This is step 2 of the GitHub OAuth flow.  The code is short-lived and can
    only be used once; GitHub returns an error if it has already been consumed.

    Args:
        code: The short-lived ``code`` parameter returned by GitHub’s OAuth
              callback redirect.

    Returns:
        The access token string on success, or ``None`` if the exchange fails
        (e.g. the code was already used, or the client credentials are wrong).
    """
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
    logger.error(
        "Failed to retrieve authenticated GitHub user — status=%d: %s",
        status, payload.get("message", ""),
    )
    return None


def get_repo_info(token: str, owner: str, repo: str) -> tuple[int, dict]:
    """Fetch repository metadata from the GitHub REST API.

    Args:
        token: GitHub access token for authentication.
        owner: Repository owner login (user or organisation).
        repo:  Repository name.

    Returns:
        ``(status_code, payload)`` from ``GET /repos/{owner}/{repo}``.
    """
    return _request("GET", f"{_API}/repos/{owner}/{repo}", token=token)


def can_push(token: str, owner: str, repo: str) -> tuple[bool, str | None, str | None]:
    """Check whether the authenticated user has push access to the given repository.

    Args:
        token: GitHub access token.
        owner: Repository owner login.
        repo:  Repository name.

    Returns:
        A three-tuple ``(can_push, default_branch, reason)``:

        - ``can_push``       is ``True`` when the user has ``push``,
                             ``admin``, or ``maintain`` permission.
        - ``default_branch`` is the repo’s default branch name, or ``None``
                             when the API call failed.
        - ``reason``         is a human-readable explanation when
                             ``can_push`` is ``False``, or ``None`` on success.
    """
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
    """Open a pull request on GitHub via the REST API.

    Args:
        token: GitHub access token with ``repo`` scope.
        owner: Repository owner login.
        repo:  Repository name.
        head:  Branch that contains the proposed changes.
        base:  Target branch to merge into (e.g. ``'main'``).
        title: PR title string.
        body:  PR description in Markdown.

    Returns:
        ``(status_code, payload)`` from the GitHub Pulls API.  A ``201``
        status with ``payload['html_url']`` indicates the PR was created.
    """
    return _request("POST", f"{_API}/repos/{owner}/{repo}/pulls", token=token,
                    data={"title": title, "head": head, "base": base, "body": body})


# ── server-side sessions ─────────────────────────────────────────────────────

def create_session(login: str, avatar_url: str, access_token: str) -> str:
    """Persist a server-side OAuth session and return a new session ID.

    The session ID is a random URL-safe token stored in an HttpOnly cookie on
    the client.  The access token never leaves the server.

    Args:
        login:        The authenticated user’s GitHub login handle.
        avatar_url:   URL of the user’s GitHub avatar image.
        access_token: The GitHub OAuth access token to store server-side.

    Returns:
        A URL-safe random session ID (32 bytes of entropy, base64url-encoded).
    """
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
    """Look up an active session by its cookie value.

    Args:
        sid: Session ID from the request cookie, or ``None`` if absent.

    Returns:
        A dict with keys ``'login'``, ``'avatar_url'``, and
        ``'access_token'``, or ``None`` if no matching session exists.
    """
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
    """Delete a session from the database (called on logout).

    Args:
        sid: Session ID to delete.  ``None`` is a silent no-op.
    """
    if not sid:
        return
    db = SessionLocal()
    try:
        db.query(GitHubSession).filter(GitHubSession.id == sid).delete()
        db.commit()
    finally:
        db.close()
