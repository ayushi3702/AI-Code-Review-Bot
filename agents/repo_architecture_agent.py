"""Repo-wide architecture agent — the analysis a diff-only bot can never do.

This agent reasons about the codebase as a whole:
  * structural smells from the file/module map (god modules, missing layering,
    inconsistent organisation),
  * cross-file duplication, found cheaply via the vector store (a chunk whose
    nearest neighbour lives in a *different* file at near-zero cosine distance
    is almost certainly copy-paste),
  * coupling / circular-import risk surfaced from import-heavy chunks.
"""
from __future__ import annotations
import datetime
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from core.config import get_chat_llm
from core.state import ScanState, ScanFinding
from core.database import SessionLocal, AgentRun
from scanner.vector_store import VectorStore
from agents._scan_common import render_context, parse_findings, _JSON_CONTRACT

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a software architect reviewing a whole repository. Identify "
    "architecture-level problems that only appear when looking across files: "
    "duplicated logic spread across modules, dead/unused modules, circular or "
    "tangled imports, modules that do too much (low cohesion), tight coupling, "
    "and inconsistent patterns/conventions between similar files. Use the file "
    "map and the duplication hints provided."
)

# A near-zero cosine distance between chunks in different files ≈ duplication.
_DUP_DISTANCE_THRESHOLD = 0.08
_MAX_DUP_PROBES = 40


def _build_repo_map(file_index: list[str], languages: dict[str, int]) -> str:
    """Format the repository’s file tree and language breakdown as a prompt string.

    Provides the architecture agent with a high-level structural overview that
    it cannot derive from individual code chunks alone.  The listing is capped
    at 300 entries to keep the prompt token count bounded.

    Args:
        file_index: Sorted list of repo-relative file paths.
        languages:  Mapping of language label → file count from the crawl stage.

    Returns:
        A two-section string: ``Languages (files per language): ...`` followed
        by an indented ``File map:`` listing.
    """
    lang_line = ", ".join(f"{k}:{v}" for k, v in sorted(languages.items()))
    listing = "\n".join(f"  {p}" for p in sorted(file_index)[:300])
    return f"Languages (files per language): {lang_line}\n\nFile map:\n{listing}"


def _find_duplicates(store: VectorStore) -> list[str]:
    """Probe a sample of chunks for near-identical twins in other files.

    Uses the vector store’s cosine similarity to find chunk pairs whose
    distance is below :data:`_DUP_DISTANCE_THRESHOLD`.  A near-zero cosine
    distance between chunks in *different* files is a strong signal of
    copy-pasted logic.

    The probe is sampled (step = ``total_chunks // _MAX_DUP_PROBES``) rather
    than exhaustive to keep latency bounded on large repos.

    Args:
        store: The :class:`~scanner.vector_store.VectorStore` for the current scan.

    Returns:
        A list of human-readable hint strings like
        ``"Near-duplicate code: foo.py ↔ bar.py"``, capped at 20 pairs.
    """
    try:
        raw = store._collection.get(include=["documents", "metadatas"])
    except Exception as e:
        logger.warning(
            "architecture agent: could not retrieve chunks for duplication analysis: %s", e,
        )
        return []

    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    hints: list[str] = []
    seen_pairs: set[frozenset] = set()

    step = max(1, len(docs) // _MAX_DUP_PROBES)
    for i in range(0, len(docs), step):
        content = docs[i]
        src_file = metas[i].get("file_path", "")
        if len(content.strip()) < 80:
            continue  # too small to call duplication meaningfully
        for rc in store.similar_to(content, top_k=2):
            if rc.file_path and rc.file_path != src_file and rc.distance <= _DUP_DISTANCE_THRESHOLD:
                pair = frozenset({src_file, rc.file_path})
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                hints.append(f"Near-duplicate code: {src_file} ↔ {rc.file_path}")
        if len(hints) >= 20:
            break
    return hints


async def repo_architecture_agent(state: ScanState) -> dict:
    """LangGraph node: run the architecture specialist agent for one scan.

    Unlike the other agents, this one cannot use the generic
    :func:`~agents._scan_common.run_retrieval_agent` helper because it needs to
    supply additional structural context — a file-tree map and duplication hints
    — that are built from repository-level data rather than from retrieval
    phrases alone.

    Extra context injected into the prompt:

    - **Repo map** — language counts and a full file listing (capped at 300
      entries) to help the model reason about module structure and organisation.
    - **Duplication hints** — file pairs whose chunks have near-zero cosine
      distance in the vector store, indicating likely copy-pasted code.
    - **Coupling chunks** — code retrieved for import, dependency, and
      global-state phrases to expose tight coupling and circular imports.

    Args:
        state: Current :class:`~core.state.ScanState` with an indexed vector
               store populated by the indexing stage.

    Returns:
        Dict with ``'findings'`` and ``'completed_agents'`` keys for LangGraph.
    """
    started = datetime.datetime.utcnow()
    logger.info(
        "architecture agent started — scan_id=%s files=%d",
        state.scan_id, len(state.file_index),
    )
    db = SessionLocal()
    run = AgentRun(scan_id=state.scan_id, agent_name="architecture", started_at=started)
    db.add(run)
    db.commit()

    try:
        store = VectorStore(state.collection_name)
        repo_map = _build_repo_map(state.file_index, state.languages)
        dup_hints = _find_duplicates(store)
        logger.info(
            "architecture agent: found %d duplication hints — scan_id=%s",
            len(dup_hints), state.scan_id,
        )
        coupling_chunks = store.query_many([
            "import module dependency package",
            "global state shared singleton configuration",
            "class inheritance base abstract interface",
        ])

        human = (
            f"{repo_map}\n\n"
            f"Duplication hints (from vector similarity):\n"
            + ("\n".join(dup_hints) if dup_hints else "  (none detected)")
            + "\n\nRepresentative import/coupling code:\n"
            + render_context(coupling_chunks)
        )

        llm = get_chat_llm()
        response = await llm.ainvoke([
            SystemMessage(content=f"{_SYSTEM}\n\n{_JSON_CONTRACT}"),
            HumanMessage(content=human),
        ])
        findings = parse_findings("architecture", response.content)

        run.status = "done"
        run.finding_count = len(findings)
        run.finished_at = datetime.datetime.utcnow()
        run.duration_ms = int((run.finished_at - started).total_seconds() * 1000)
        db.commit()

        logger.info(
            "architecture agent completed — %d findings (%d dup hints) in %dms — scan_id=%s",
            len(findings), len(dup_hints), run.duration_ms, state.scan_id,
        )
        return {"findings": findings, "completed_agents": ["architecture"]}

    except Exception as e:
        run.status = "failed"
        db.commit()
        logger.error(
            "architecture agent failed — scan_id=%s: %s",
            state.scan_id, e, exc_info=True,
        )
        return {"errors": [f"architecture: {e}"]}
    finally:
        db.close()
