"""Repo-wide performance agent.

Pulls chunks involving loops, queries, I/O and data structures, then asks the
model to flag algorithmic and resource inefficiencies across the codebase.
"""
from __future__ import annotations

from core.state import ScanState
from agents._scan_common import run_retrieval_agent

_PHRASES = [
    "for loop while loop nested iteration",
    "database query call inside loop N+1",
    "blocking io synchronous request inside async function",
    "large list comprehension memory allocation unbounded growth",
    "sort search algorithm complexity O(n^2)",
    "cache memoization repeated computation",
    "pagination limit offset missing on query",
    "file read whole file into memory streaming",
]

_SYSTEM = (
    "You are a performance engineer reviewing an entire repository. Identify "
    "algorithmic inefficiencies (quadratic work where linear is possible), N+1 "
    "query patterns, blocking I/O inside async code, unbounded memory growth, "
    "missing pagination/limits, and repeated expensive computation that should be "
    "cached. Focus on issues that would actually degrade latency or throughput."
)


async def repo_performance_agent(state: ScanState) -> dict:
    """LangGraph node: run the performance specialist agent for one scan.

    Retrieves code chunks related to loops, database queries, I/O patterns, and
    data-structure usage, then asks the model to flag algorithmic inefficiencies,
    N+1 query patterns, blocking I/O inside async code, unbounded memory growth,
    missing pagination, and repeated expensive computation that should be cached.

    Args:
        state: Current :class:`~core.state.ScanState` with an indexed vector
               store populated by the indexing stage.

    Returns:
        Dict with ``'findings'`` and ``'completed_agents'`` keys for LangGraph.
    """
    return await run_retrieval_agent("performance", _SYSTEM, _PHRASES, state)
