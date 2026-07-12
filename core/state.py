"""Shared Pydantic state models for the repo-wide deep-scan pipeline.

ScanFinding  — a single issue raised by one specialist agent.
ScanState    — the mutable pipeline state that flows through LangGraph.
               LangGraph passes this object from node to node; the
               ``operator.add`` reducer on list fields lets the four parallel
               agents merge their findings and errors safely without
               overwriting each other.
"""
from __future__ import annotations
from typing import Annotated
from pydantic import BaseModel, Field
import operator


# ── Repo-wide deep-scan state (full-repository review mode) ───────────────────

class ScanFinding(BaseModel):
    """A single finding produced by one specialist review agent.

    Attributes:
        agent:          Which agent raised the issue — ``'security'``,
                        ``'performance'``, ``'architecture'``, or ``'quality'``.
        severity:       Impact level — ``'high'``, ``'medium'``, or ``'low'``.
        file:           Repo-relative path to the affected source file.
        line:           Optional 1-based line number within the file.
        title:          Short headline shown in the report summary.
        detail:         Full explanation of why the issue matters.
        recommendation: Concrete, actionable fix suggestion.
        code_snippet:   Optional code excerpt illustrating the problem.
    """
    agent: str             # security | performance | architecture | quality
    severity: str          # high | medium | low
    file: str
    line: int | None = None
    title: str
    detail: str
    recommendation: str = ""
    code_snippet: str | None = None


class ScanState(BaseModel):
    """Mutable pipeline state shared across all LangGraph nodes for one scan.

    Fields are populated in three stages:

    1. **Indexing**  — ``scan_id``, ``repo_source``, ``repo_name``,
                       ``file_count``, ``chunk_count``, ``languages``,
                       ``file_index``, ``collection_name``.
    2. **Analysis**  — ``findings``, ``completed_agents``, ``errors``.
                       All three use ``operator.add`` so the four parallel
                       agents can append without overwriting each other.
    3. **Reporting** — ``score``, ``grade``, ``report_markdown``,
                       ``report_html``.
    """
    # Input / identity
    scan_id: str = ""
    repo_source: str = ""             # URL or local path
    repo_name: str = ""
    collection_name: str = ""         # ChromaDB collection for this scan

    # Index stats (filled by the crawl/index stage before agents fan out)
    file_count: int = 0
    chunk_count: int = 0
    languages: dict[str, int] = Field(default_factory=dict)
    file_index: list[str] = Field(default_factory=list)   # repo-relative paths

    # Agent outputs — merged via addition since agents run in parallel
    findings: Annotated[list[ScanFinding], operator.add] = Field(default_factory=list)

    # Final report
    score: int = 0                    # 0-100 health score
    grade: str = ""                   # A–F
    report_markdown: str = ""
    report_html: str = ""

    # Bookkeeping
    completed_agents: Annotated[list[str], operator.add] = Field(default_factory=list)
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)
