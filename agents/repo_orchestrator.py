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
    """No-op LangGraph node that acts as the fan-out entry point.

    LangGraph requires an explicit source node before it can branch to the four
    parallel specialist agents.  This node performs no computation — it simply
    returns an empty dict so the graph can dispatch to all four branches.
    """
    return {}  # no-op fan-out point


def build_scan_graph():
    """Construct and compile the LangGraph scan pipeline.

    The graph has the following topology::

        fanout ──┬── security ──┐
               ├── performance ─┪── report ── END
               ├── architecture ┪┘
               └── quality ────┘

    The four specialist agents run concurrently; their findings are merged into
    shared state via the ``operator.add`` reducer on
    :attr:`~core.state.ScanState.findings`.  The report node only fires once
    all four agent edges arrive.

    Returns:
        A compiled :class:`langgraph.graph.StateGraph` ready to be invoked.
    """
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
    """Update arbitrary columns on a :class:`~core.database.Scan` row in-place.

    Opens a short-lived database session, applies the given field updates, and
    commits.  Used throughout the pipeline to track stage and status changes
    without passing a database session through the LangGraph state.

    Args:
        scan_id: The scan ID whose row should be updated.
        **fields: Column name / value pairs to set on the row.
    """
    db = SessionLocal()
    try:
        scan = db.query(Scan).filter(Scan.id == scan_id).first()
        if scan:
            for k, v in fields.items():
                setattr(scan, k, v)
            db.commit()
        else:
            logger.warning("_update_scan: scan row not found — scan_id=%s", scan_id)
    except Exception as e:
        logger.error("_update_scan failed — scan_id=%s: %s", scan_id, e)
    finally:
        db.close()


async def index_repo(state: ScanState, progress=None) -> ScanState:
    """Crawl, chunk, and embed the repository into ChromaDB.

    This is the first (sequential) stage of the pipeline.  It must complete
    before the four parallel analysis agents are launched because they all
    query the shared vector store built here.

    Pipeline steps:
    1. Clone or verify the local repo via :func:`~scanner.crawler.acquire_repo`.
    2. Walk the file tree with :func:`~scanner.crawler.crawl`, filtering out
       vendored code, binaries, and lock files.
    3. Chunk source files along semantic boundaries with
       :func:`~scanner.chunker.chunk_files`.
    4. Embed chunks and store them in a per-scan ChromaDB collection via
       :class:`~scanner.vector_store.VectorStore`.

    Args:
        state:    The current :class:`~core.state.ScanState` (mutated in-place).
        progress: Optional callable that accepts a status message string; used
                  by the CLI to update the spinner text.

    Returns:
        The updated :class:`~core.state.ScanState` with ``file_count``,
        ``chunk_count``, ``languages``, and ``file_index`` populated.
    """
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
    """Run a full end-to-end repository scan and return the populated state.

    Entry point shared by both the FastAPI server (background task) and the
    CLI.  Orchestrates the complete pipeline:

    1. Ensures a :class:`~core.database.Scan` row exists in the database.
    2. Calls :func:`index_repo` to crawl, chunk, and embed the codebase.
    3. Invokes the compiled LangGraph pipeline which fans out to the four
       specialist agents in parallel, then fans back in to the report agent.
    4. Persists the final report and score to the database.

    Args:
        repo_source: GitHub repository URL to analyse.
        scan_id:     Optional pre-assigned scan ID; a UUID is generated if
                     omitted.
        progress:    Optional callable for status messages (used by the CLI).

    Returns:
        The fully populated :class:`~core.state.ScanState` including findings,
        Markdown/HTML reports, score, and grade.  On failure the ``errors``
        field is populated and ``status`` is set to ``'failed'``.
    """
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

    logger.info("Scan started — scan_id=%s repo=%s", scan_id, repo_source)
    try:
        state = await index_repo(state, progress=progress)

        if state.chunk_count == 0:
            logger.warning(
                "No source files found — aborting — scan_id=%s repo=%s",
                scan_id, repo_source,
            )
            _update_scan(scan_id, status="failed", stage="done",
                         error="No source files to analyze.")
            return state

        logger.info(
            "Launching parallel agents (security, performance, architecture, quality) — scan_id=%s",
            scan_id,
        )
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
        logger.info(
            "Scan completed — scan_id=%s repo=%s score=%d grade=%s findings=%d",
            scan_id, repo_source, result.score, result.grade, len(result.findings),
        )
        if progress:
            progress(f"Done — score {result.score}/100 (grade {result.grade})")
        return result

    except Exception as e:
        logger.error(
            "Scan failed — scan_id=%s repo=%s: %s",
            scan_id, repo_source, e, exc_info=True,
        )
        _update_scan(scan_id, status="failed", stage="done", error=str(e))
        state.errors.append(str(e))
        return state
