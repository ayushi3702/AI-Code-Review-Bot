from __future__ import annotations
from typing import Annotated
from pydantic import BaseModel, Field
import operator


# ── Repo-wide deep-scan state (full-repository review mode) ───────────────────

class ScanFinding(BaseModel):
    agent: str             # security | performance | architecture | quality
    severity: str          # high | medium | low
    file: str
    line: int | None = None
    title: str
    detail: str
    recommendation: str = ""
    code_snippet: str | None = None


class ScanState(BaseModel):
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
