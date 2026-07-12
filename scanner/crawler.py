"""Repo crawler: turn a GitHub URL into a clean list of source files.

Two jobs:
  1. Acquire the code — `git clone` the GitHub URL into a temp workspace.
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
    """Return ``True`` if ``source`` looks like a remote Git URL.

    Accepts ``http://``, ``https://``, and ``git@`` prefixes, as well as any
    string ending with ``.git``.

    Args:
        source: Repository source string to test.
    """
    return source.startswith(("http://", "https://", "git@")) or source.endswith(".git")


def acquire_repo(source: str, scan_id: str) -> tuple[str, str, bool]:
    """Obtain a local copy of the repository to crawl.

    Clones the remote GitHub URL into
    ``<SCAN_WORKSPACE_DIR>/<scan_id>/`` with ``depth=1`` (shallow clone) to
    minimise download time.  Any pre-existing directory at that path is removed
    before cloning to guarantee a clean working tree.

    Args:
        source:  GitHub repository URL (HTTPS or SSH).
        scan_id: Scan ID used as the destination subdirectory name.

    Returns:
        A three-tuple ``(local_root, repo_name, is_temp_clone)``:

        - ``local_root``    — absolute path to the cloned directory.
        - ``repo_name``     — repository name derived from the URL
                             (last path component, ``.git`` suffix removed).
        - ``is_temp_clone`` — always ``True`` for remote URLs; used by the
                             orchestrator to decide whether to delete the clone
                             after indexing.

    Raises:
        ValueError: If ``source`` is not a recognised remote Git URL.
    """
    if not _is_git_url(source):
        raise ValueError(
            "Only GitHub repository URLs are supported (e.g. "
            "https://github.com/owner/repo.git)."
        )

    from git import Repo

    dest = os.path.join(SCAN_WORKSPACE_DIR, scan_id)
    os.makedirs(SCAN_WORKSPACE_DIR, exist_ok=True)
    if os.path.exists(dest):
        shutil.rmtree(dest, ignore_errors=True)
    logger.info("Cloning %s → %s (depth=1)", source, dest)
    try:
        Repo.clone_from(source, dest, depth=1)
    except Exception as e:
        logger.error("Failed to clone repository %s: %s", source, e, exc_info=True)
        raise
    repo_name = source.rstrip("/").split("/")[-1].removesuffix(".git")
    return dest, repo_name, True


def _load_gitignore(root: str) -> pathspec.PathSpec | None:
    """Load and parse the ``.gitignore`` file at the repository root.

    Args:
        root: Absolute path to the repository root directory.

    Returns:
        A compiled :class:`pathspec.PathSpec` on success, or ``None`` if no
        ``.gitignore`` file exists or it cannot be read.
    """
    gi = os.path.join(root, ".gitignore")
    if not os.path.isfile(gi):
        return None
    try:
        with open(gi, "r", encoding="utf-8", errors="ignore") as f:
            return pathspec.PathSpec.from_lines("gitwildmatch", f.readlines())
    except Exception as e:
        logger.warning("Could not load .gitignore from %s: %s", gi, e)
        return None


def crawl(root: str) -> list[SourceFile]:
    """Walk the repository tree and return every source file worth indexing.

    Applies several layers of filtering to avoid polluting the vector store
    with noise:

    - **Directory exclusions** — ``node_modules``, ``.git``, ``__pycache__``,
      ``dist``, ``build``, and other well-known non-source directories.
    - **Extension filter** — only files with extensions in :data:`EXT_LANGUAGE`
      are included (Python, JS/TS, Go, Java, Ruby, PHP, C/C++, Rust, etc.).
    - **Pattern exclusions** — lock files, minified bundles, source maps,
      generated protobuf stubs, and TypeScript declaration files.
    - **Gitignore** — files matched by the repo’s ``.gitignore`` are skipped.
    - **Size gate** — files larger than :data:`~core.config.MAX_FILE_SIZE_KB`
      KB or empty files are skipped.
    - **Encoding** — files that cannot be decoded as UTF-8 (binary or exotic
      encodings) are skipped.

    Args:
        root: Absolute path to the repository root.

    Returns:
        List of :class:`SourceFile` objects ready for chunking.
    """
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
            if size == 0:
                continue
            if size > max_bytes:
                logger.warning(
                    "Skipping oversized file %s (%d KB > limit %d KB)",
                    rel_path, size // 1024, MAX_FILE_SIZE_KB,
                )
                continue

            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except (UnicodeDecodeError, OSError) as e:
                logger.warning("Skipping unreadable file %s: %s", rel_path, e)
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
    """Remove the temporary clone directory created by :func:`acquire_repo`.

    Called by the orchestrator after the vector store is fully populated, since
    the agents only query the vector store and no longer need the raw files.
    Uses ``ignore_errors=True`` so a partial or missing directory does not
    raise an exception.

    Args:
        scan_id: The scan ID whose clone directory should be deleted.
    """
    dest = os.path.join(SCAN_WORKSPACE_DIR, scan_id)
    shutil.rmtree(dest, ignore_errors=True)
