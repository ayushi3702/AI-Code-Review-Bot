"""Tests for the repo-wide scanner's deterministic pieces.

These exercise the crawler's filtering and the chunker's boundary detection —
no Azure/OpenAI calls, so they run offline in CI.
"""
from __future__ import annotations
import os

from scanner.crawler import crawl, EXT_LANGUAGE, EXCLUDED_DIRS
from scanner.chunker import chunk_file, chunk_files
from scanner.crawler import SourceFile


# ── Crawler ───────────────────────────────────────────────────────────────────

def test_crawl_picks_source_skips_junk(tmp_path):
    (tmp_path / "app.py").write_text("def hello():\n    return 1\n", encoding="utf-8")
    (tmp_path / "util.js").write_text("export function add(a,b){return a+b}\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# docs\n", encoding="utf-8")  # not a code ext
    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    (node_modules / "dep.js").write_text("module.exports = 1\n", encoding="utf-8")
    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")

    files = crawl(str(tmp_path))
    paths = {f.path for f in files}

    assert "app.py" in paths
    assert "util.js" in paths
    assert "README.md" not in paths          # md not in EXT_LANGUAGE
    assert "node_modules/dep.js" not in paths  # excluded dir
    assert "package-lock.json" not in paths    # excluded file pattern


def test_crawl_detects_languages(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.ts").write_text("const x = 1\n", encoding="utf-8")
    files = {f.path: f.language for f in crawl(str(tmp_path))}
    assert files["a.py"] == "python"
    assert files["b.ts"] == "typescript"


def test_excluded_dirs_contains_common_vendors():
    for d in ("node_modules", ".git", "venv", "dist"):
        assert d in EXCLUDED_DIRS


# ── Chunker ───────────────────────────────────────────────────────────────────

def _sf(path, lang, content):
    return SourceFile(path=path, abs_path=path, language=lang, content=content, size_bytes=len(content))


def test_python_chunker_splits_by_function():
    code = (
        "import os\n\n"
        "def first():\n    return 1\n\n"
        "def second():\n    return 2\n\n"
        "class Thing:\n    def method(self):\n        return 3\n"
    )
    chunks = chunk_file(_sf("m.py", "python", code))
    symbols = {c.symbol for c in chunks}
    assert "first" in symbols
    assert "second" in symbols
    assert "Thing" in symbols


def test_generic_chunker_detects_js_functions():
    code = (
        "export function alpha() { return 1; }\n"
        "function beta() { return 2; }\n"
        "const gamma = () => 3;\n"
    )
    chunks = chunk_file(_sf("m.js", "javascript", code))
    assert len(chunks) >= 2
    symbols = {c.symbol for c in chunks if c.symbol}
    assert "alpha" in symbols or "beta" in symbols


def test_chunk_files_returns_chunks_with_line_ranges():
    code = "def only():\n    return 1\n"
    chunks = chunk_files([_sf("x.py", "python", code)])
    assert chunks
    c = chunks[0]
    assert c.start_line >= 1
    assert c.end_line >= c.start_line
    assert c.file_path == "x.py"
