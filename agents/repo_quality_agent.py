"""Repo-wide code-quality agent.

Covers correctness/maintainability concerns that aren't security or performance:
error handling, dead code, missing tests, confusing naming, and risky patterns.
"""
from __future__ import annotations

from core.state import ScanState
from agents._scan_common import run_retrieval_agent

_PHRASES = [
    "exception handling try except catch swallow error bare except",
    "TODO FIXME hack temporary workaround",
    "duplicated logic copy paste repeated code",
    "magic number hardcoded constant configuration",
    "function too long complex deeply nested conditionals",
    "missing return value none null handling edge case",
    "logging print debug statement left in code",
    "test assertion coverage missing unit test",
]

_SYSTEM = (
    "You are a staff engineer reviewing overall code quality and correctness across "
    "a repository. Flag: swallowed/over-broad exception handling, likely bugs and "
    "unhandled edge cases, dead or unreachable code, copy-pasted logic that should be "
    "extracted, overly complex functions, and important code paths that lack tests. "
    "Be specific and actionable; skip pure formatting opinions."
)


async def repo_quality_agent(state: ScanState) -> dict:
    return await run_retrieval_agent("quality", _SYSTEM, _PHRASES, state)
