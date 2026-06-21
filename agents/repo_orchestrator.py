"""Repo-wide deep-scan orchestrator.

Pipeline:

    acquire + crawl + chunk + embed  (indexing — sequential, must finish first)
                 │
                 ▼
             ┌── security ──────┐
             ├── performance ───┤   ← LangGraph fan-out (run concurrently)
   index ────┼── architecture ──┼──► report ──► END   (fan-in)
             └── quality ───────┘

The four specialist agents share nothing but the vector store, so LangGraph runs
them in parallel; their findings merge into shared state via the `operator.add`
reducer on ScanState.findings. The report node only fires once all four finish.
"""
from __future__ import annotations
import uuid
import logging
import datetime

from langgraph.graph import StateGraph, END

from core.state import ScanState
from core.database import SessionLocal, Scan, init_db
from scanner.crawler import acquire_repo, crawl, cleanup_clone
from scanner.chunker import chunk_files
from scanner.vector_store import VectorStore
from agents.repo_security_agent import repo_security_agent
from agents.repo_performance_agent import repo_performance_agent
from agents.repo_architecture_agent import repo_architecture_agent
from agents.repo_quality_agent import repo_quality_agent
from agents.repo_report_agent import repo_report_agent

logger = logging.getLogger(__name__)


async def _fanout_node(state: ScanState) -> dict:
    return {}  # no-op fan-out point


def build_scan_graph():
    g = StateGraph(ScanState)
    g.add_node("fanout", _fanout_node)
    g.add_node("security", repo_security_agent)
    g.add_node("performance", repo_performance_agent)
    g.add_node("architecture", repo_architecture_agent)
    g.add_node("quality", repo_quality_agent)
    g.add_node("report", repo_report_agent)

    g.set_entry_point("fanout")
    for agent in ("security", "performance", "architecture", "quality"):
        g.add_edge("fanout", agent)
        g.add_edge(agent, "report")
    g.add_edge("report", END)
    return g.compile()


scan_pipeline = build_scan_graph()


def _update_scan(scan_id: str, **fields) -> None:
    db = SessionLocal()
    try:
        scan = db.query(Scan).filter(Scan.id == scan_id).first()
        if scan:
            for k, v in fields.items():
                setattr(scan, k, v)
            db.commit()
    finally:
        db.close()


async def index_repo(state: ScanState, progress=None) -> ScanState:
    """Crawl → chunk → embed into ChromaDB. Mutates and returns state."""
    def _p(msg):
        if progress:
            progress(msg)
        logger.info(msg)

    _update_scan(state.scan_id, status="running", stage="crawl")
    local_root, repo_name, is_temp = acquire_repo(state.repo_source, state.scan_id)
    state.repo_name = repo_name
    _p(f"Acquired repo '{repo_name}'")

    files = crawl(local_root)
    state.file_count = len(files)
    state.file_index = [f.path for f in files]
    langs: dict[str, int] = {}
    for f in files:
        langs[f.language] = langs.get(f.language, 0) + 1
    state.languages = langs
    _p(f"Crawled {len(files)} source files")

    _update_scan(state.scan_id, stage="index", file_count=len(files))
    chunks = chunk_files(files)
    state.chunk_count = len(chunks)
    _p(f"Chunked into {len(chunks)} semantic units — embedding…")

    store = VectorStore(state.collection_name)
    store.index(chunks)
    _update_scan(state.scan_id, stage="analyze", chunk_count=len(chunks))
    _p(f"Indexed {len(chunks)} chunks into ChromaDB")

    if is_temp:
        # we keep the clone until after analysis isn't needed — agents read the
        # vector store, not the files — so it's safe to remove now.
        cleanup_clone(state.scan_id)
    return state


async def run_scan(repo_source: str, scan_id: str | None = None, progress=None) -> ScanState:
    """Full end-to-end scan. Returns the populated ScanState (with reports)."""
    init_db()
    scan_id = scan_id or str(uuid.uuid4())
    collection = f"scan_{scan_id.replace('-', '')[:24]}"

    # ensure a Scan row exists
    db = SessionLocal()
    if not db.query(Scan).filter(Scan.id == scan_id).first():
        db.add(Scan(id=scan_id, repo_source=repo_source, status="queued", stage="queued"))
        db.commit()
    db.close()

    state = ScanState(
        scan_id=scan_id,
        repo_source=repo_source,
        collection_name=collection,
    )

    try:
        state = await index_repo(state, progress=progress)

        if state.chunk_count == 0:
            _update_scan(scan_id, status="failed", stage="done",
                         error="No source files to analyze.")
            return state

        if progress:
            progress("Running security · performance · architecture · quality agents…")
        final = await scan_pipeline.ainvoke(state)

        # LangGraph returns a dict-like; rebuild a ScanState for the caller
        result = ScanState(**{**state.model_dump(), **{
            "findings": final.get("findings", []),
            "report_markdown": final.get("report_markdown", ""),
            "report_html": final.get("report_html", ""),
            "score": final.get("score", 0),
            "grade": final.get("grade", ""),
            "errors": final.get("errors", []),
        }})
        if progress:
            progress(f"Done — score {result.score}/100 (grade {result.grade})")
        return result

    except Exception as e:
        logger.exception("Scan failed")
        _update_scan(scan_id, status="failed", stage="done", error=str(e))
        state.errors.append(str(e))
        return state
