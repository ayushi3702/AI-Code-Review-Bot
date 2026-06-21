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
    lang_line = ", ".join(f"{k}:{v}" for k, v in sorted(languages.items()))
    listing = "\n".join(f"  {p}" for p in sorted(file_index)[:300])
    return f"Languages (files per language): {lang_line}\n\nFile map:\n{listing}"


def _find_duplicates(store: VectorStore) -> list[str]:
    """Probe a sample of chunks for near-identical twins in other files."""
    try:
        raw = store._collection.get(include=["documents", "metadatas"])
    except Exception:
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
    started = datetime.datetime.utcnow()
    db = SessionLocal()
    run = AgentRun(scan_id=state.scan_id, agent_name="architecture", started_at=started)
    db.add(run)
    db.commit()

    try:
        store = VectorStore(state.collection_name)
        repo_map = _build_repo_map(state.file_index, state.languages)
        dup_hints = _find_duplicates(store)
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

        logger.info("[architecture] produced %d findings (%d dup hints)", len(findings), len(dup_hints))
        return {"findings": findings, "completed_agents": ["architecture"]}

    except Exception as e:
        run.status = "failed"
        db.commit()
        logger.error("[architecture] %s", e)
        return {"errors": [f"architecture: {e}"]}
    finally:
        db.close()
