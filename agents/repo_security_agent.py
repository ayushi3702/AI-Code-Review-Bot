"""Repo-wide security agent.

Retrieves the chunks most likely to contain security-sensitive logic
(auth, input handling, queries, secrets, network calls) and asks the model to
flag real vulnerabilities across the whole codebase — not just a diff.
"""
from __future__ import annotations

from core.state import ScanState
from agents._scan_common import run_retrieval_agent

_PHRASES = [
    "authentication login password token session",
    "user input request parameters form data validation",
    "SQL query database execute raw string",
    "hardcoded secret api key credential private key",
    "subprocess shell command os.system exec eval",
    "file path open read write upload deserialization pickle",
    "http request url fetch ssrf redirect external call",
    "encryption hashing jwt verify signature",
]

_SYSTEM = (
    "You are a senior application security engineer auditing an entire repository. "
    "Identify concrete vulnerabilities: injection (SQL/command/template), broken "
    "authentication or authorization, hardcoded secrets, insecure deserialization, "
    "SSRF, path traversal, weak crypto, unsafe use of eval/exec, and missing input "
    "validation at trust boundaries. Prioritise exploitable issues over theoretical ones."
)


async def repo_security_agent(state: ScanState) -> dict:
    """LangGraph node: run the security specialist agent for one scan.

    Retrieves code chunks that are most likely to contain security-sensitive
    logic (authentication, input handling, SQL queries, secrets, network calls,
    file I/O, deserialization) and asks the model to identify exploitable
    vulnerabilities — injection flaws, broken auth, hardcoded secrets, SSRF,
    path traversal, weak crypto, and unsafe use of ``eval``/``exec``.

    Args:
        state: Current :class:`~core.state.ScanState` with an indexed vector
               store populated by the indexing stage.

    Returns:
        Dict with ``'findings'`` and ``'completed_agents'`` keys for LangGraph.
    """
