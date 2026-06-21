"""Shared plumbing for the repo-wide scan agents.

Every specialist agent does the same dance: pull concern-relevant chunks from
the vector store, pack them into a single prompt, ask Azure OpenAI for
structured findings, and parse the JSON back into ScanFinding objects. This
module centralises that so each agent file only declares *what* it cares about
(its retrieval phrases + system prompt), not *how* to call the model.
"""
from __future__ import annotations
import json
import datetime
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from core.config import get_chat_llm
from core.state import ScanState, ScanFinding
from core.database import SessionLocal, AgentRun
from scanner.vector_store import VectorStore, RetrievedChunk

logger = logging.getLogger(__name__)

# Keep the prompt bounded regardless of repo size.
MAX_CHUNKS_IN_PROMPT = 18
MAX_CHARS_PER_CHUNK = 1500

_JSON_CONTRACT = (
    'Respond ONLY with a JSON object of this exact shape:\n'
    '{"findings": [{"severity": "high|medium|low", "file": "<path>", '
    '"line": <int or null>, "title": "<short>", "detail": "<why it matters>", '
    '"recommendation": "<concrete fix>"}]}\n'
    "Return an empty array if you find nothing. Never invent files that are not "
    "shown to you. Report only real, actionable issues — no style nitpicks."
)


def render_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for c in chunks[:MAX_CHUNKS_IN_PROMPT]:
        sym = f" symbol={c.symbol}" if c.symbol else ""
        body = c.content[:MAX_CHARS_PER_CHUNK]
        parts.append(
            f"--- File: {c.file_path} (lines {c.start_line}-{c.end_line}){sym} ---\n{body}"
        )
    return "\n\n".join(parts)


def parse_findings(agent: str, raw: str) -> list[ScanFinding]:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("[%s] could not parse model JSON", agent)
        return []
    out: list[ScanFinding] = []
    for f in data.get("findings", []):
        if not f.get("title") or not f.get("file"):
            continue
        out.append(ScanFinding(
            agent=agent,
            severity=str(f.get("severity", "medium")).lower(),
            file=f.get("file", ""),
            line=f.get("line"),
            title=f.get("title", ""),
            detail=f.get("detail", ""),
            recommendation=f.get("recommendation", ""),
        ))
    return out


async def run_retrieval_agent(
    agent_name: str,
    system_prompt: str,
    retrieval_phrases: list[str],
    state: ScanState,
    extra_human: str = "",
) -> dict:
    """Generic agent body: retrieve → prompt → parse → record."""
    started = datetime.datetime.utcnow()
    db = SessionLocal()
    run = AgentRun(scan_id=state.scan_id, agent_name=agent_name, started_at=started)
    db.add(run)
    db.commit()

    try:
        store = VectorStore(state.collection_name)
        chunks = store.query_many(retrieval_phrases)
        if not chunks and not extra_human:
            run.status = "done"
            run.finding_count = 0
            run.finished_at = datetime.datetime.utcnow()
            db.commit()
            return {"findings": [], "completed_agents": [agent_name]}

        context = render_context(chunks)
        human = f"{extra_human}\n\nCode under review:\n{context}" if extra_human else (
            f"Code under review:\n{context}"
        )

        llm = get_chat_llm()
        response = await llm.ainvoke([
            SystemMessage(content=f"{system_prompt}\n\n{_JSON_CONTRACT}"),
            HumanMessage(content=human),
        ])
        findings = parse_findings(agent_name, response.content)

        run.status = "done"
        run.finding_count = len(findings)
        run.finished_at = datetime.datetime.utcnow()
        run.duration_ms = int((run.finished_at - started).total_seconds() * 1000)
        db.commit()

        logger.info("[%s] produced %d findings", agent_name, len(findings))
        return {"findings": findings, "completed_agents": [agent_name]}

    except Exception as e:
        run.status = "failed"
        db.commit()
        logger.error("[%s] %s", agent_name, e)
        return {"errors": [f"{agent_name}: {e}"]}
    finally:
        db.close()
