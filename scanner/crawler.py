"""Repo crawler: turn a GitHub URL or local path into a clean list of source files.

Two jobs:
  1. Acquire the code — `git clone` a URL into a temp workspace, or read a local dir.
  2. Filter aggressively — skip vendored deps, binaries, lockfiles, generated code.
     This protects the embedding budget: indexing `node_modules` is pure waste.
"""
from __future__ import annotations
import os
import shutil
import logging
from dataclasses import dataclass

import pathspec

from core.config import SCAN_WORKSPACE_DIR, MAX_FILE_SIZE_KB

logger = logging.getLogger(__name__)


# Map file extensions → a language label the agents can reason about.
EXT_LANGUAGE = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".rs": "rust",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
    ".kt": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
    ".sql": "sql",
    ".sh": "shell", ".bash": "shell",
}

# Directories we never descend into.
EXCLUDED_DIRS = {
    ".git", "node_modules", "venv", ".venv", "env", "__pycache__",
    "dist", "build", "out", "target", ".next", ".nuxt", "vendor",
    ".idea", ".vscode", "coverage", ".pytest_cache", ".mypy_cache",
    "bin", "obj", ".chroma", ".scan_workspace", "site-packages",
}

# Filename patterns we skip even when the extension looks like source.
EXCLUDED_FILE_PATTERNS = [
    "*.lock", "*.min.js", "*.min.css", "*-lock.json", "package-lock.json",
    "yarn.lock", "poetry.lock", "Pipfile.lock", "*.map", "*.snap",
    "*.generated.*", "*_pb2.py", "*.d.ts",
]


@dataclass
class SourceFile:
    path: str          # repo-relative path, forward slashes
    abs_path: str
    language: str
    content: str
    size_bytes: int


def _is_git_url(source: str) -> bool:
    return source.startswith(("http://", "https://", "git@")) or source.endswith(".git")


def acquire_repo(source: str, scan_id: str) -> tuple[str, str, bool]:
    """Return (local_root, repo_name, is_temp_clone).

    If `source` is a URL we clone it under SCAN_WORKSPACE_DIR/<scan_id>.
    If it's a local path we use it in place (is_temp_clone=False).
    """
    if _is_git_url(source):
        from git import Repo

        dest = os.path.join(SCAN_WORKSPACE_DIR, scan_id)
        os.makedirs(SCAN_WORKSPACE_DIR, exist_ok=True)
        if os.path.exists(dest):
            shutil.rmtree(dest, ignore_errors=True)
        logger.info("Cloning %s → %s", source, dest)
        Repo.clone_from(source, dest, depth=1)
        repo_name = source.rstrip("/").split("/")[-1].removesuffix(".git")
        return dest, repo_name, True

    local_root = os.path.abspath(source)
    if not os.path.isdir(local_root):
        raise ValueError(f"Local path is not a directory: {local_root}")
    repo_name = os.path.basename(local_root.rstrip(os.sep)) or "repo"
    return local_root, repo_name, False


def _load_gitignore(root: str) -> pathspec.PathSpec | None:
    gi = os.path.join(root, ".gitignore")
    if not os.path.isfile(gi):
        return None
    try:
        with open(gi, "r", encoding="utf-8", errors="ignore") as f:
            return pathspec.PathSpec.from_lines("gitwildmatch", f.readlines())
    except Exception:
        return None


def crawl(root: str) -> list[SourceFile]:
    """Walk `root` and return the source files worth indexing."""
    gitignore = _load_gitignore(root)
    skip_spec = pathspec.PathSpec.from_lines("gitwildmatch", EXCLUDED_FILE_PATTERNS)
    max_bytes = MAX_FILE_SIZE_KB * 1024

    files: list[SourceFile] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded dirs in place so os.walk doesn't descend into them
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS and not d.startswith(".")]

        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            language = EXT_LANGUAGE.get(ext)
            if not language:
                continue

            abs_path = os.path.join(dirpath, name)
            rel_path = os.path.relpath(abs_path, root).replace(os.sep, "/")

            if skip_spec.match_file(rel_path) or skip_spec.match_file(name):
                continue
            if gitignore and gitignore.match_file(rel_path):
                continue

            try:
                size = os.path.getsize(abs_path)
            except OSError:
                continue
            if size > max_bytes or size == 0:
                continue

            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except (UnicodeDecodeError, OSError):
                continue  # binary or unreadable — skip

            files.append(SourceFile(
                path=rel_path,
                abs_path=abs_path,
                language=language,
                content=content,
                size_bytes=size,
            ))

    logger.info("Crawled %d source files under %s", len(files), root)
    return files


def cleanup_clone(scan_id: str) -> None:
    dest = os.path.join(SCAN_WORKSPACE_DIR, scan_id)
    shutil.rmtree(dest, ignore_errors=True)
