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
    """Format a list of retrieved code chunks into a single prompt-ready string.

    Each chunk is rendered as a fenced block with a header line that shows the
    file path, line range, and (if available) the enclosing symbol name.  The
    output is truncated at :data:`MAX_CHUNKS_IN_PROMPT` chunks and each chunk’s
    body is capped at :data:`MAX_CHARS_PER_CHUNK` characters to keep prompt
    token usage bounded regardless of repo size.

    Args:
        chunks: Retrieved chunks from the vector store, ordered by relevance.

    Returns:
        A single string of ``--- File: ... ---`` blocks joined by blank lines,
        ready to be embedded in an LLM prompt.
    """
    parts = []
    for c in chunks[:MAX_CHUNKS_IN_PROMPT]:
        sym = f" symbol={c.symbol}" if c.symbol else ""
        body = c.content[:MAX_CHARS_PER_CHUNK]
        parts.append(
            f"--- File: {c.file_path} (lines {c.start_line}-{c.end_line}){sym} ---\n{body}"
        )
    return "\n\n".join(parts)


def parse_findings(agent: str, raw: str) -> list[ScanFinding]:
    """Parse the LLM’s JSON response into a list of :class:`~core.state.ScanFinding` objects.

    The model is instructed to return a JSON object of the form
    ``{"findings": [{...}, ...]}``.  Any finding entry that is missing a
    ``title`` or ``file`` key is silently dropped to avoid surfacing incomplete
    results in the report.

    Args:
        agent: Agent name tag to attach to every parsed finding.
        raw:   Raw string response from the LLM.

    Returns:
        A (possibly empty) list of :class:`~core.state.ScanFinding` objects.
        Returns ``[]`` and logs a warning if the response is not valid JSON.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "%s agent: could not parse LLM response as JSON — response length=%d chars",
            agent, len(raw or ""),
        )
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
    """Generic agent body shared by the security, performance, and quality agents.

    Performs the standard retrieve → prompt → parse → record cycle:

    1. Queries the scan’s ChromaDB vector store with each phrase in
       ``retrieval_phrases`` and merges the results (de-duplicated by chunk
       identity).
    2. Packs up to :data:`MAX_CHUNKS_IN_PROMPT` chunks into a single LLM prompt
       together with ``system_prompt`` and the JSON contract.
    3. Parses the response into :class:`~core.state.ScanFinding` objects via
       :func:`parse_findings`.
    4. Records an :class:`~core.database.AgentRun` row with timing and finding
       count for observability.

    Args:
        agent_name:        Short identifier used in logs and as the
                           :attr:`~core.state.ScanFinding.agent` tag.
        system_prompt:     Domain-specific instructions for the LLM.
        retrieval_phrases: Natural-language concern phrases used to pull the
                           most relevant code chunks from the vector store.
        state:             Current :class:`~core.state.ScanState`.
        extra_human:       Optional additional text prepended to the human
                           turn (used by the architecture agent to include the
                           repo map and duplication hints).

    Returns:
        A dict with ``'findings'`` and ``'completed_agents'`` keys on success,
        or an ``'errors'`` key on failure — both compatible with LangGraph’s
        state-merge reducer.
    """
    started = datetime.datetime.utcnow()
    logger.info("%s agent started — scan_id=%s", agent_name, state.scan_id)
    db = SessionLocal()
    run = AgentRun(scan_id=state.scan_id, agent_name=agent_name, started_at=started)
    db.add(run)
    db.commit()

    try:
        store = VectorStore(state.collection_name)
        chunks = store.query_many(retrieval_phrases)
        if not chunks and not extra_human:
            logger.warning(
                "%s agent: no relevant chunks found in vector store — scan_id=%s",
                agent_name, state.scan_id,
            )
            run.status = "done"
            run.finding_count = 0
            run.finished_at = datetime.datetime.utcnow()
            db.commit()
            return {"findings": [], "completed_agents": [agent_name]}

        logger.info(
            "%s agent: retrieved %d chunks from vector store — invoking LLM — scan_id=%s",
            agent_name, len(chunks), state.scan_id,
        )

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

        logger.info(
            "%s agent completed — %d findings in %dms — scan_id=%s",
            agent_name, len(findings), run.duration_ms, state.scan_id,
        )
        return {"findings": findings, "completed_agents": [agent_name]}

    except Exception as e:
        run.status = "failed"
        db.commit()
        logger.error(
            "%s agent failed — scan_id=%s: %s",
            agent_name, state.scan_id, e, exc_info=True,
        )
        return {"errors": [f"{agent_name}: {e}"]}
    finally:
        db.close()
