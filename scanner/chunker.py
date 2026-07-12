"""Chunk source files along semantic boundaries (functions/classes), not blind
line windows.

Why boundary-aware chunks matter: when the security agent retrieves "the chunk
about authentication", we want a whole function, not lines 40-80 that cut a
function in half. For Python we use the real AST. For every other language we
fall back to a lightweight regex that detects declaration lines (def/function/
class/func/public ...) and splits there, with a hard size cap so a giant file
never produces one monster chunk.
"""
from __future__ import annotations
import ast
import re
import hashlib
import logging
from dataclasses import dataclass

from scanner.crawler import SourceFile

logger = logging.getLogger(__name__)

# Soft target: keep chunks well under the embedding model's input limit while
# staying large enough to carry real context.
MAX_CHUNK_LINES = 120
MIN_CHUNK_LINES = 4


@dataclass
class Chunk:
    chunk_id: str
    file_path: str
    language: str
    symbol: str | None     # function/class name if we could detect one
    start_line: int        # 1-based
    end_line: int
    content: str


def _make_id(file_path: str, start_line: int, content: str) -> str:
    """Generate a stable, unique chunk ID from its location and content.

    The ID is a ``<file_path>#<start_line>-<hash>`` string where ``<hash>`` is
    the first 12 hex characters of a SHA-1 over the file path, start line, and
    first 64 bytes of content.  This makes IDs deterministic across re-indexing
    the same file, which helps ChromaDB avoid duplicate embeddings.

    Args:
        file_path:  Repo-relative path to the source file.
        start_line: 1-based line number where the chunk begins.
        content:    The chunk’s text content.

    Returns:
        A string of the form ``"path/to/file.py#10-a3f2c1d8e9b0"``.
    """
    h = hashlib.sha1(f"{file_path}:{start_line}:{content[:64]}".encode()).hexdigest()[:12]
    return f"{file_path}#{start_line}-{h}"


# ── Python: real AST chunking ─────────────────────────────────────────────────

def _chunk_python(sf: SourceFile) -> list[Chunk]:
    """Chunk a Python source file using the real AST for accurate boundaries.

    Walks the module-level AST nodes and emits one chunk per top-level
    function, async function, or class definition.  Decorator lines that
    precede the definition are included in the chunk so the full context is
    preserved.  Any module-level code not covered by a top-level definition
    (imports, constants, ``if __name__ == '__main__'`` blocks) is handled by
    the generic chunker fallback.

    Falls back to :func:`_chunk_generic` when the file contains a syntax error
    (e.g. Python 2 files or partially valid code).

    Args:
        sf: The :class:`~scanner.crawler.SourceFile` to chunk.

    Returns:
        List of :class:`Chunk` objects covering the file’s top-level symbols.
    """
    try:
        tree = ast.parse(sf.content)
    except SyntaxError as e:
        logger.warning(
            "Python AST parse failed for %s — falling back to generic chunker: %s",
            sf.path, e,
        )
        return _chunk_generic(sf)

    lines = sf.content.split("\n")
    chunks: list[Chunk] = []
    covered_end = 0

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start) or start
            # capture leading decorators
            if node.decorator_list:
                start = min(start, node.decorator_list[0].lineno)
            body = "\n".join(lines[start - 1:end])
            chunks.append(Chunk(
                chunk_id=_make_id(sf.path, start, body),
                file_path=sf.path,
                language=sf.language,
                symbol=node.name,
                start_line=start,
                end_line=end,
                content=body,
            ))
            covered_end = max(covered_end, end)

    # module-level code that isn't inside a class/func (imports, constants, glue)
    if not chunks:
        return _chunk_generic(sf)
    return _split_oversized(chunks)


# ── Generic: regex declaration detection ──────────────────────────────────────

_DECL_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:public\s+|private\s+|protected\s+|static\s+|async\s+)*"
    r"(?:function\s+\w+|func\s+\w+|class\s+\w+|interface\s+\w+|type\s+\w+|def\s+\w+|"
    r"(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\(|\w+\s*:\s*(?:async\s*)?\()"
)
_NAME_RE = re.compile(r"(?:function|func|class|interface|type|def|const|let|var)\s+(\w+)")


def _chunk_generic(sf: SourceFile) -> list[Chunk]:
    """Chunk a non-Python source file using a regex to detect declaration boundaries.

    Scans for lines that look like function, class, interface, or variable
    declaration starts (``function foo``, ``def bar``, ``class Baz``,
    ``const fn = (``, etc.) using :data:`_DECL_RE`.  Each matched line becomes
    the start of a new chunk; the previous chunk ends just before it.

    Falls back to :func:`_window_chunks` for files with no detectable
    declarations (SQL scripts, shell scripts, configuration files).

    Args:
        sf: The :class:`~scanner.crawler.SourceFile` to chunk.

    Returns:
        List of :class:`Chunk` objects aligned to declaration boundaries.
    """
    lines = sf.content.split("\n")
    # find declaration boundary line numbers (0-based)
    boundaries = [i for i, ln in enumerate(lines) if _DECL_RE.match(ln)]
    if not boundaries:
        return _window_chunks(sf, lines)

    boundaries = sorted(set(boundaries))
    if boundaries[0] != 0:
        boundaries.insert(0, 0)
    boundaries.append(len(lines))

    chunks: list[Chunk] = []
    for i in range(len(boundaries) - 1):
        start_idx, end_idx = boundaries[i], boundaries[i + 1]
        block = lines[start_idx:end_idx]
        if len("\n".join(block).strip()) == 0:
            continue
        m = _NAME_RE.search(lines[start_idx])
        chunks.append(Chunk(
            chunk_id=_make_id(sf.path, start_idx + 1, "\n".join(block)),
            file_path=sf.path,
            language=sf.language,
            symbol=m.group(1) if m else None,
            start_line=start_idx + 1,
            end_line=end_idx,
            content="\n".join(block),
        ))
    return _split_oversized(chunks)


def _window_chunks(sf: SourceFile, lines: list[str]) -> list[Chunk]:
    """Fallback chunker: split a file into fixed-size line windows.

    Used for files with no detectable declaration boundaries (config files,
    SQL scripts, shell scripts).  Windows are :data:`MAX_CHUNK_LINES` lines
    each; empty windows are skipped.

    Args:
        sf:    The source file being chunked.
        lines: Pre-split list of lines from the file content.

    Returns:
        List of fixed-size :class:`Chunk` objects.
    """
    chunks = []
    for start in range(0, len(lines), MAX_CHUNK_LINES):
        block = lines[start:start + MAX_CHUNK_LINES]
        if not "\n".join(block).strip():
            continue
        chunks.append(Chunk(
            chunk_id=_make_id(sf.path, start + 1, "\n".join(block)),
            file_path=sf.path,
            language=sf.language,
            symbol=None,
            start_line=start + 1,
            end_line=start + len(block),
            content="\n".join(block),
        ))
    return chunks


def _split_oversized(chunks: list[Chunk]) -> list[Chunk]:
    """Break any chunk that exceeds :data:`MAX_CHUNK_LINES` into smaller windows.

    A top-level class with hundreds of methods may produce a single very large
    chunk from the AST or regex pass.  This post-processor splits such chunks
    into :data:`MAX_CHUNK_LINES`-line sub-chunks so no single embedding input
    is too large for the model’s token limit.

    Args:
        chunks: Output from the primary chunking pass.

    Returns:
        New list where every chunk is at most :data:`MAX_CHUNK_LINES` lines.
    """
    out: list[Chunk] = []
    for c in chunks:
        lines = c.content.split("\n")
        if len(lines) <= MAX_CHUNK_LINES:
            out.append(c)
            continue
        for off in range(0, len(lines), MAX_CHUNK_LINES):
            block = lines[off:off + MAX_CHUNK_LINES]
            s = c.start_line + off
            out.append(Chunk(
                chunk_id=_make_id(c.file_path, s, "\n".join(block)),
                file_path=c.file_path,
                language=c.language,
                symbol=c.symbol,
                start_line=s,
                end_line=s + len(block) - 1,
                content="\n".join(block),
            ))
    return out


def chunk_file(sf: SourceFile) -> list[Chunk]:
    """Dispatch a single source file to the appropriate chunker.

    Python files are chunked via the real AST (:func:`_chunk_python`); all
    other languages use the regex-based declaration detector
    (:func:`_chunk_generic`).

    Args:
        sf: The :class:`~scanner.crawler.SourceFile` to chunk.

    Returns:
        List of :class:`Chunk` objects ready for embedding.
    """
    if sf.language == "python":
        return _chunk_python(sf)
    return _chunk_generic(sf)


def chunk_files(files: list[SourceFile]) -> list[Chunk]:
    """Chunk all source files in the crawled repository.

    Iterates over every :class:`~scanner.crawler.SourceFile`, delegates to
    :func:`chunk_file`, and concatenates the results into a single flat list.

    Args:
        files: List of source files returned by :func:`~scanner.crawler.crawl`.

    Returns:
        Flat list of all :class:`Chunk` objects from the entire repository,
        ready to be embedded and stored by
        :class:`~scanner.vector_store.VectorStore`.
    """
    chunks: list[Chunk] = []
    for sf in files:
        chunks.extend(chunk_file(sf))
    logger.info(
        "Chunked %d source files into %d semantic units",
        len(files), len(chunks),
    )
    return chunks
