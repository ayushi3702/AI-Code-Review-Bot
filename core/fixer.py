"""Turn findings into concrete, committable code changes.

Two responsibilities:
  1. generate_fix()  — ask the model for an exact (original → replacement)
     snippet for one finding, persist it, and return a unified-diff preview.
  2. apply_and_commit() — take a set of generated fixes, apply them to the
     working tree, detect overlapping edits (our "merge conflicts"), verify the
     result still parses (and an optional verify command passes), then stage +
     commit the survivors in a single git commit — and optionally push it.

Nothing is committed or pushed unless the batch is conflict-free AND the patched
files still pass validation. If the edits would break the code, the working tree
is restored and the operation is aborted.

Committing only works for *local-path* scans: the repo lives on disk at
`scan.repo_source`. Remote-URL scans clone into a throwaway workspace that is
deleted after indexing, and pushing back would need credentials, so those are
reported as not committable.
"""
from __future__ import annotations
import os
import json
import uuid
import shutil
import difflib
import logging
import subprocess

from langchain_core.messages import SystemMessage, HumanMessage

from core.config import get_chat_llm, SCAN_WORKSPACE_DIR
from core.database import SessionLocal, Scan, ScanFindingRow, ScanFix
from core import github_auth

logger = logging.getLogger(__name__)

MAX_FILE_CHARS = 16000  # cap what we feed the model so huge files stay in budget

# Optional project-level check (tests/build/lint) run before committing. If set,
# a non-zero exit means the patched code is "breaking" and the batch is aborted.
VERIFY_CMD = os.getenv("CODE_REVIEW_VERIFY_CMD", "").strip()

_FIX_SYSTEM = (
    "You are a senior engineer triaging a single review finding. First decide "
    "whether the finding can be resolved by an ACTUAL code change, or whether the "
    "remedy is only ADVISORY.\n\n"
    "Return `kind`:\n"
    '- "fix": you can resolve the finding by editing real, executable code. Provide '
    "a small contiguous snippet copied VERBATIM from the file as `original_code`, "
    "and its corrected version as `suggested_code`.\n"
    '- "suggestion": the remedy is advisory, architectural, or process-level '
    "(e.g. \"adopt Alembic\", \"add tests\", \"run migrations out-of-band\") and "
    "cannot be expressed as a safe, concrete code replacement. In this case return "
    "EMPTY strings for both `original_code` and `suggested_code` — do NOT invent "
    "edits, do NOT just add a comment, and do NOT remove any lines. Put the advice "
    "in `explanation`.\n\n"
    "Rules for a \"fix\":\n"
    "- `original_code` MUST appear character-for-character in the file (same "
    "indentation, same whitespace) so it can be located exactly.\n"
    "- Keep the snippet as small as possible while still being unique in the file.\n"
    "- Change only what is needed to fix the finding; preserve surrounding style.\n"
    "- The change must alter actual code — never a comment-only edit. If the only "
    'thing you would add is a comment, classify it as a "suggestion" instead.\n'
    "- If no safe automatic code fix is possible, use \"suggestion\".\n\n"
    'Respond ONLY with JSON: {"kind": "fix"|"suggestion", '
    '"original_code": "<verbatim or empty>", '
    '"suggested_code": "<replacement or empty>", "explanation": "<one line>"}'
)


# Line-comment prefixes for the languages we commonly touch. Used to detect
# "comment-only" edits that don't actually change executable code.
_COMMENT_PREFIXES = ("#", "//", "--", "/*", "*", "*/", ";", "<!--", "-->")


def _strip_noise(code: str) -> str:
    """Return ``code`` with blank lines and pure-comment lines removed.

    Used by :func:`_is_comment_only_change` to detect whether two code snippets
    differ only in comments or whitespace, which would make a proposed fix
    non-committable (a comment-only edit is classified as a suggestion instead).

    Args:
        code: Source code string to strip.

    Returns:
        Multi-line string with blank and comment-only lines removed.
    """
    out = []
    for line in code.splitlines():
        s = line.strip()
        if not s or s.startswith(_COMMENT_PREFIXES):
            continue
        out.append(s)
    return "\n".join(out)


def _is_comment_only_change(original: str, suggested: str) -> bool:
    """Return ``True`` if ``original`` and ``suggested`` differ only in comments or whitespace.

    A change that only adds, removes, or rewrites comments without touching
    executable code is not a safe automatic fix — it would silently overwrite
    existing comments and confuse reviewers.  Such changes are reclassified as
    advisory suggestions.

    Args:
        original:  The verbatim code snippet from the file.
        suggested: The model-proposed replacement snippet.

    Returns:
        ``True`` when stripping comments and blank lines from both sides
        produces identical text.
    """
    return _strip_noise(original) == _strip_noise(suggested)


def _window_around(content: str, line: int | None, budget: int) -> tuple[str, int, int, int]:
    """Return a char-budgeted excerpt of `content` centered on `line`.

    Large files can't be sent whole, and slicing the first N chars often cuts
    off the very code a finding refers to (e.g. line 1583). Instead we grow a
    window outward from the target line until we hit the budget, so the model
    actually sees the relevant code and can produce a verbatim replacement.

    Returns (excerpt, start_line, end_line, total_lines), all 1-based.
    """
    total = content.count("\n") + 1
    if len(content) <= budget:
        return content, 1, total, total
    lines = content.splitlines()
    n = len(lines)
    idx = max(0, min(n - 1, (line or 1) - 1))
    start = end = idx
    size = len(lines[idx]) + 1
    while True:
        grew = False
        if start > 0 and size + len(lines[start - 1]) + 1 <= budget:
            start -= 1
            size += len(lines[start]) + 1
            grew = True
        if end < n - 1 and size + len(lines[end + 1]) + 1 <= budget:
            end += 1
            size += len(lines[end]) + 1
            grew = True
        if not grew:
            break
    return "\n".join(lines[start:end + 1]), start + 1, end + 1, n


def _is_git_url(source: str) -> bool:
    """Return ``True`` if ``source`` looks like a remote Git URL.

    Accepts ``http://``, ``https://``, ``git@`` prefixes, and any string ending
    with ``.git``.

    Args:
        source: The repository source string to test.
    """
    return source.startswith(("http://", "https://", "git@")) or source.endswith(".git")


def ensure_worktree(scan: Scan, token: str | None = None) -> str:
    """Clone the scan's GitHub repo into a worktree so fixes can be committed.

    The repo is cloned once (authenticated with the user's token) into the scan
    workspace, then reused, so fixes can be applied, committed and pushed back.
    """
    if not _is_git_url(scan.repo_source):
        raise ValueError("Only GitHub repositories are supported for committing.")
    if not token:
        raise ValueError("Sign in with GitHub to apply fixes to a repository.")
    parsed = github_auth.parse_github_repo(scan.repo_source)
    if not parsed:
        raise ValueError("Only GitHub remote repositories are supported for committing.")
    owner, repo = parsed

    dest = os.path.join(SCAN_WORKSPACE_DIR, f"work_{scan.id}")
    if os.path.isdir(os.path.join(dest, ".git")):
        return dest

    from git import Repo
    os.makedirs(SCAN_WORKSPACE_DIR, exist_ok=True)
    auth_url = f"https://x-access-token:{token}@github.com/{owner}/{repo}.git"
    logger.info("Cloning %s/%s for commit worktree into %s", owner, repo, dest)
    try:
        Repo.clone_from(auth_url, dest)
    except Exception as e:
        logger.error("Failed to clone %s/%s for commit worktree: %s", owner, repo, e, exc_info=True)
        raise
    return dest


def _build_diff(file: str, original: str, suggested: str) -> str:
    """Build a unified-diff preview string for a single file change.

    Args:
        file:      Repo-relative path used as the diff header label.
        original:  The exact verbatim snippet that will be replaced.
        suggested: The replacement snippet.

    Returns:
        A unified diff string (``--- a/file`` / ``+++ b/file`` format) suitable
        for display in the UI code-review panel.
    """
    a = original.splitlines(keepends=True)
    b = suggested.splitlines(keepends=True)
    diff = difflib.unified_diff(a, b, fromfile=f"a/{file}", tofile=f"b/{file}")
    return "".join(diff)


# extensions we can cheaply syntax-check before committing
_NODE_EXTS = {".js", ".mjs", ".cjs"}


def _validate_files(root: str, files: list[str]) -> list[dict]:
    """Return a list of breakages for patched files that no longer parse.

    Best-effort, language-aware syntax checks only — we never commit a change
    that turns a previously-parseable file into a syntactically broken one.
    """
    breaks: list[dict] = []
    node = shutil.which("node")
    for rel in files:
        abs_path = os.path.join(root, rel)
        ext = os.path.splitext(rel)[1].lower()
        try:
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except OSError as e:
            breaks.append({"file": rel, "reason": f"cannot read file: {e}"})
            continue
        if ext == ".py":
            try:
                compile(text, abs_path, "exec")
            except SyntaxError as e:
                logger.warning(
                    "Syntax validation: Python syntax error in %s at line %d: %s",
                    rel, e.lineno, e.msg,
                )
                breaks.append({"file": rel, "reason": f"Python syntax error: {e.msg} (line {e.lineno})"})
        elif ext in _NODE_EXTS and node:
            proc = subprocess.run([node, "--check", abs_path],
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                msg = (proc.stderr or proc.stdout or "syntax error").strip().splitlines()[-1:]
                logger.warning(
                    "Syntax validation: JavaScript syntax error in %s: %s",
                    rel, "".join(msg),
                )
                breaks.append({"file": rel, "reason": f"JS syntax error: {''.join(msg)}"})
    return breaks


def _run_verify_cmd(root: str) -> str | None:
    """Run the optional project verify command. Return an error string if it fails."""
    if not VERIFY_CMD:
        return None
    try:
        proc = subprocess.run(VERIFY_CMD, shell=True, cwd=root,
                              capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        logger.warning("Verify command timed out after 600s: %s", VERIFY_CMD)
        return f"verify command timed out: {VERIFY_CMD}"
    except Exception as e:  # noqa: BLE001 - surface any launch failure
        logger.error("Verify command could not launch — %s: %s", VERIFY_CMD, e)
        return f"verify command could not run: {e}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
        logger.warning(
            "Verify command failed (exit %d): %s",
            proc.returncode, VERIFY_CMD,
        )
        return f"verify command failed (exit {proc.returncode}):\n" + "\n".join(tail)
    return None


def _fix_to_dict(fix: ScanFix) -> dict:
    return {
        "finding_id": fix.finding_id,
        "file": fix.file,
        "original_code": fix.original_code,
        "suggested_code": fix.suggested_code,
        "explanation": fix.explanation,
        "diff": fix.diff,
        "status": fix.status,
        "commit_sha": fix.commit_sha,
        "kind": "suggestion" if fix.status == "suggestion" else "fix",
        "applicable": fix.status == "ready" and bool(fix.original_code),
    }


async def generate_fix(scan_id: str, finding_id: str, token: str | None = None) -> dict:
    """Generate (or return a cached) concrete code fix for one finding.

    Asks the LLM to produce an exact ``(original_code, suggested_code)`` pair
    for the finding’s file.  The result is classified as:

    - ``'ready'``        — the snippet was located in the file and changes real
                           executable code; safe to commit.
    - ``'suggestion'``   — the model’s remedy is advisory or comment-only;
                           displayed to the developer but not committable.
    - ``'unapplicable'`` — the snippet could not be found in the file (e.g. the
                           file changed since the scan).

    Args:
        scan_id:    ID of the owning scan.
        finding_id: ID of the :class:`~core.database.ScanFindingRow` to fix.
        token:      Optional GitHub OAuth token; required when the repo must be
                    cloned from a remote URL to read the source file.

    Returns:
        A dict representation of the :class:`~core.database.ScanFix` row,
        including ``status``, ``diff``, ``explanation``, and ``applicable``.

    Raises:
        ValueError: If the finding or scan row does not exist, or if the
                    source file cannot be found in the repo.
    """
    db = SessionLocal()
    try:
        existing = (
            db.query(ScanFix)
            .filter(ScanFix.finding_id == finding_id, ScanFix.status != "committed")
            .first()
        )
        if existing:
            logger.info(
                "generate_fix: returning cached fix — finding_id=%s status=%s",
                finding_id, existing.status,
            )
            return _fix_to_dict(existing)

        finding = db.query(ScanFindingRow).filter(ScanFindingRow.id == finding_id).first()
        if not finding:
            raise ValueError("finding not found")
        scan = db.query(Scan).filter(Scan.id == scan_id).first()
        if not scan:
            raise ValueError("scan not found")

        logger.info(
            "generate_fix: scan_id=%s finding_id=%s file=%s title=%r",
            scan_id, finding_id, finding.file, finding.title,
        )

        root = ensure_worktree(scan, token)
        abs_path = os.path.join(root, finding.file)
        if not os.path.isfile(abs_path):
            raise ValueError(f"file not found in repo: {finding.file}")

        with open(abs_path, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()

        excerpt, win_start, win_end, total_lines = _window_around(
            content, finding.line, MAX_FILE_CHARS
        )
        location = (
            f"File `{finding.file}` (full, {total_lines} lines)"
            if win_start == 1 and win_end == total_lines
            else f"File `{finding.file}` (lines {win_start}-{win_end} of {total_lines})"
        )
        human = (
            f"Finding: {finding.title}\n"
            f"Severity: {finding.severity}\n"
            f"Why it matters: {finding.detail}\n"
            f"Suggested direction: {finding.recommendation}\n"
            f"Line hint: {finding.line or 'n/a'}\n\n"
            f"{location}:\n```\n{excerpt}\n```"
        )
        llm = get_chat_llm()
        resp = await llm.ainvoke([
            SystemMessage(content=_FIX_SYSTEM),
            HumanMessage(content=human),
        ])
        try:
            data = json.loads(resp.content)
        except (json.JSONDecodeError, TypeError):
            data = {}

        original = (data.get("original_code") or "").strip("\n")
        suggested = data.get("suggested_code") or ""
        explanation = data.get("explanation") or ""
        kind = (data.get("kind") or "").strip().lower()

        # A fix is committable only when it locates real code AND actually
        # changes executable code.  Advisory output, or edits that merely
        # add/rewrite comments without touching code, are non-committable.
        located = bool(original) and original in content
        comment_only = located and _is_comment_only_change(original, suggested)
        is_fix = kind != "suggestion" and located and not comment_only

        if is_fix:
            status = "ready"
            diff = _build_diff(finding.file, original, suggested)
            logger.info(
                "generate_fix: fix ready — finding_id=%s file=%s",
                finding_id, finding.file,
            )
        else:
            status = "suggestion" if (kind == "suggestion" or comment_only) else "unapplicable"
            if comment_only:
                logger.warning(
                    "generate_fix: model proposed comment-only edit — classified as suggestion — finding_id=%s",
                    finding_id,
                )
            elif not located:
                logger.warning(
                    "generate_fix: snippet not found in file — fix unapplicable — finding_id=%s file=%s",
                    finding_id, finding.file,
                )
            else:
                logger.warning(
                    "generate_fix: fix unapplicable (kind=%s) — finding_id=%s",
                    kind, finding_id,
                )
            # Never carry a code edit for a non-committable suggestion.
            original = ""
            suggested = ""
            diff = ""

        fix = ScanFix(
            scan_id=scan_id,
            finding_id=finding_id,
            file=finding.file,
            original_code=original,
            suggested_code=suggested,
            explanation=explanation,
            diff=diff,
            status=status,
        )
        db.add(fix)
        db.commit()
        db.refresh(fix)
        return _fix_to_dict(fix)
    finally:
        db.close()


def apply_and_commit(scan_id: str, finding_ids: list[str], message: str,
                     mode: str = "pr", token: str | None = None,
                     login: str | None = None) -> dict:
    """Apply the selected fixes and commit them together.

    Safety gates, in order — the commit only happens if all pass:
      1. No conflicts (overlapping edits / stale or ambiguous snippets).
      2. The patched files still parse (syntax check) and the optional verify
         command succeeds — otherwise the working tree is restored and nothing
         is committed.

    `mode` controls what happens after a clean commit:
      * "direct" — push the commit to the repo's branch on origin.
      * "pr"     — commit on a fresh branch, push it, and open a Pull Request.
    A push/PR failure leaves the commit intact locally and is reported.
    """
    from git import Repo, Actor
    from git.exc import InvalidGitRepositoryError, GitCommandError

    db = SessionLocal()
    created_branch = None
    base_branch = None
    repo = None
    try:
        scan = db.query(Scan).filter(Scan.id == scan_id).first()
        if not scan:
            raise ValueError("scan not found")

        logger.info(
            "apply_and_commit: scan_id=%s user=%s fixes=%d mode=%s",
            scan_id, login or "anonymous", len(finding_ids), mode,
        )

        root = ensure_worktree(scan, token)

        try:
            repo = Repo(root)
        except InvalidGitRepositoryError:
            raise ValueError(f"Not a git repository: {root}")

        # figure out the GitHub coordinates (for push/PR auth + the PR API call)
        gh_owner = gh_repo = None
        parsed = github_auth.parse_github_repo(scan.repo_source)
        if parsed:
            gh_owner, gh_repo = parsed

        try:
            base_branch = repo.active_branch.name
        except TypeError:
            base_branch = None  # detached HEAD

        # ── pre-flight checks for push/PR so we never touch the tree in vain ──
        if mode in ("direct", "pr"):
            if not base_branch:
                raise ValueError("repository is in a detached HEAD state — cannot push")
            if mode == "pr" and not (gh_owner and token):
                raise ValueError("opening a Pull Request requires GitHub sign-in")

        fixes = (
            db.query(ScanFix)
            .filter(ScanFix.finding_id.in_(finding_ids), ScanFix.scan_id == scan_id)
            .all()
        )
        found_ids = {f.finding_id for f in fixes}
        missing = [fid for fid in finding_ids if fid not in found_ids]

        conflicts: list[dict] = []
        by_file: dict[str, list[tuple[ScanFix, int, int]]] = {}
        for fx in fixes:
            if not fx.original_code:
                conflicts.append({"finding_id": fx.finding_id, "file": fx.file,
                                  "reason": "no applicable automatic fix"})
                continue
            abs_path = os.path.join(root, fx.file)
            if not os.path.isfile(abs_path):
                conflicts.append({"finding_id": fx.finding_id, "file": fx.file,
                                  "reason": "file missing"})
                continue
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
            idx = content.find(fx.original_code)
            if idx == -1:
                conflicts.append({"finding_id": fx.finding_id, "file": fx.file,
                                  "reason": "code changed since scan — snippet no longer matches"})
                continue
            if content.find(fx.original_code, idx + 1) != -1:
                conflicts.append({"finding_id": fx.finding_id, "file": fx.file,
                                  "reason": "snippet is ambiguous (matches multiple locations)"})
                continue
            by_file.setdefault(fx.file, []).append((fx, idx, idx + len(fx.original_code)))

        # detect overlapping edits within the same file
        for file, spans in by_file.items():
            spans.sort(key=lambda s: s[1])
            for (fa, sa, ea), (fb, sb, eb) in zip(spans, spans[1:]):
                if sb < ea:  # overlap
                    conflicts.append({
                        "finding_id": fb.finding_id, "file": file,
                        "reason": f"overlaps another selected fix ({fa.finding_id})",
                    })

        if missing:
            for fid in missing:
                conflicts.append({"finding_id": fid, "file": None,
                                  "reason": "no fix generated yet — preview it first"})

        if conflicts:
            logger.warning(
                "apply_and_commit: %d conflict(s) detected — aborting commit — scan_id=%s",
                len(conflicts), scan_id,
            )
            return {"committed": False, "pushed": False, "conflicts": conflicts,
                    "message": "Resolve conflicts (deselect the listed fixes) and try again."}

        # for PR mode, branch off the base before touching files
        if mode == "pr":
            created_branch = f"ai-code-review/{uuid.uuid4().hex[:8]}"
            repo.git.checkout("-b", created_branch)

        # apply to the working tree, keeping originals so we can roll back
        originals: dict[str, str] = {}
        changed_files: list[str] = []
        applied: list[str] = []
        for file, spans in by_file.items():
            abs_path = os.path.join(root, file)
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
            originals[abs_path] = content
            for fx, start, end in sorted(spans, key=lambda s: s[1], reverse=True):
                content = content[:start] + fx.suggested_code + content[end:]
                applied.append(fx.finding_id)
            with open(abs_path, "w", encoding="utf-8") as fh:
                fh.write(content)
            changed_files.append(file)

        # ── break-safety gate: validate before committing ───────────────────
        breaks = _validate_files(root, changed_files)
        verify_error = None if breaks else _run_verify_cmd(root)
        if breaks or verify_error:
            logger.warning(
                "apply_and_commit: syntax/verify check failed — rolling back — scan_id=%s breaks=%d",
                scan_id, len(breaks),
            )
            for abs_path, original in originals.items():
                with open(abs_path, "w", encoding="utf-8") as fh:
                    fh.write(original)
            if created_branch:
                repo.git.checkout(base_branch)
                repo.git.branch("-D", created_branch)
            return {
                "committed": False, "pushed": False, "conflicts": [],
                "breaks": breaks,
                "verify_error": verify_error,
                "message": "Fixes would break the code — reverted, nothing committed.",
            }

        repo.index.add(changed_files)
        if login:
            actor = Actor(login, f"{login}@users.noreply.github.com")
        else:
            actor = Actor("AI Code Review Bot", "bot@codereview.local")
        commit = repo.index.commit(message or "Apply AI code review fixes",
                                   author=actor, committer=actor)
        sha = commit.hexsha
        logger.info(
            "apply_and_commit: committed — scan_id=%s sha=%s files=%s mode=%s",
            scan_id, sha[:8], changed_files, mode,
        )

        for fx in fixes:
            if fx.finding_id in applied:
                fx.status = "committed"
                fx.commit_sha = sha
        db.commit()

        # ── push / open PR (only after a clean commit) ──────────────────────
        pushed = False
        push_error = None
        pr_url = None
        branch_to_push = created_branch or base_branch

        def _push(branch: str) -> None:
            """Push `branch` to GitHub (authenticated)."""
            auth_url = f"https://x-access-token:{token}@github.com/{gh_owner}/{gh_repo}.git"
            repo.git.push(auth_url, f"{branch}:{branch}")

        if mode in ("direct", "pr"):
            try:
                _push(branch_to_push)
                pushed = True
            except (GitCommandError, ValueError) as e:
                push_error = str(e)
                logger.error(
                    "apply_and_commit: push failed — scan_id=%s branch=%s: %s",
                    scan_id, branch_to_push, e,
                )

        if mode == "pr" and pushed:
            title = message or f"AI code review: {len(applied)} fix(es)"
            body = ("Automated fixes generated by the AI Code Review Platform.\n\n"
                    "Each change passed syntax validation before committing.")
            status, payload = github_auth.create_pull_request(
                token, gh_owner, gh_repo, head=created_branch, base=base_branch,
                title=title, body=body,
            )
            if status in (200, 201):
                pr_url = payload.get("html_url")
                logger.info("apply_and_commit: PR opened — %s — scan_id=%s", pr_url, scan_id)
            else:
                push_error = f"branch pushed but PR could not be opened: {payload.get('message', status)}"
                logger.warning(
                    "apply_and_commit: PR could not be opened — scan_id=%s: %s",
                    scan_id, push_error,
                )

        return {
            "committed": True,
            "pushed": pushed,
            "push_error": push_error,
            "mode": mode,
            "branch": branch_to_push,
            "pr_url": pr_url,
            "commit_sha": sha,
            "short_sha": sha[:8],
            "files": changed_files,
            "applied": applied,
            "conflicts": [],
        }
    finally:
        db.close()


