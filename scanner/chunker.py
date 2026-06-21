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
from dataclasses import dataclass

from scanner.crawler import SourceFile

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
    h = hashlib.sha1(f"{file_path}:{start_line}:{content[:64]}".encode()).hexdigest()[:12]
    return f"{file_path}#{start_line}-{h}"


# ── Python: real AST chunking ─────────────────────────────────────────────────

def _chunk_python(sf: SourceFile) -> list[Chunk]:
    try:
        tree = ast.parse(sf.content)
    except SyntaxError:
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
    """Fallback for files with no detectable declarations (config, SQL, scripts)."""
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
    """Break any chunk longer than MAX_CHUNK_LINES into line windows."""
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
    if sf.language == "python":
        return _chunk_python(sf)
    return _chunk_generic(sf)


def chunk_files(files: list[SourceFile]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sf in files:
        chunks.extend(chunk_file(sf))
    return chunks
