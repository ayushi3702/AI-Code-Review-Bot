from __future__ import annotations
import uuid
import datetime
import os
from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()


class AgentRun(Base):
    """Execution record for a single specialist agent within one scan.

    One row is created when the agent starts and updated with timing data and
    finding count when it finishes.  ``status`` transitions from ``'running'``
    to either ``'done'`` or ``'failed'``.
    """
    __tablename__ = "agent_runs"

    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    scan_id     = Column(String)            # which scan this agent run belongs to
    agent_name  = Column(String)
    status      = Column(String, default="running")
    finding_count = Column(Integer, default=0)
    duration_ms = Column(Integer)
    started_at  = Column(DateTime)
    finished_at = Column(DateTime)


# ── Repo-wide deep-scan persistence ──────────────────────────────────────────

class Scan(Base):
    """Top-level scan record tracking the full lifecycle of one repository review.

    Created with ``status='queued'`` when a scan request arrives and updated as
    the pipeline advances through stages: ``crawl → index → analyze → report``.
    Stores the final HTML and Markdown reports, health score, and A–F grade once
    the pipeline completes.  The ``error`` field captures the failure reason when
    ``status='failed'``.
    """
    __tablename__ = "scans"

    id            = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    repo_source   = Column(String, nullable=False)   # URL or local path
    repo_name     = Column(String)
    status        = Column(String, default="queued")  # queued|running|done|failed
    stage         = Column(String, default="queued")  # crawl|index|analyze|report|done
    file_count    = Column(Integer, default=0)
    chunk_count   = Column(Integer, default=0)
    finding_count = Column(Integer, default=0)
    score         = Column(Integer)
    grade         = Column(String)
    file_list     = Column(Text)        # JSON array of repo-relative file paths
    report_markdown = Column(Text)
    report_html   = Column(Text)
    error         = Column(Text)
    created_at    = Column(DateTime, default=datetime.datetime.utcnow)
    finished_at   = Column(DateTime)


class ScanFindingRow(Base):
    """Persisted finding row — one row per issue reported by an agent.

    Mirrors :class:`~core.state.ScanFinding` but lives in the database so the
    React frontend can fetch findings for a completed scan without rerunning the
    agents.  ``code_snippet`` carries an optional verbatim extract of the
    affected code for display in the UI.
    """
    __tablename__ = "scan_findings"

    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    scan_id     = Column(String, ForeignKey("scans.id"))
    agent       = Column(String)
    severity    = Column(String)
    file        = Column(String)
    line        = Column(Integer)
    title       = Column(String)
    detail      = Column(Text)
    recommendation = Column(Text)
    code_snippet = Column(Text)


class ScanFix(Base):
    """A concrete, committable code change generated for a single finding.

    ``status`` lifecycle:
    - ``'ready'``       — an applicable ``original_code`` → ``suggested_code``
                          replacement was found and a unified diff produced.
    - ``'suggestion'``  — the model's remedy is advisory (e.g. “adopt Alembic”)
                          and cannot be expressed as a safe code replacement.
    - ``'unapplicable'``— the snippet could not be located in the current file.
    - ``'committed'``   — the fix was successfully applied and committed.
    """
    __tablename__ = "scan_fixes"

    id            = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    scan_id       = Column(String, ForeignKey("scans.id"))
    finding_id    = Column(String, ForeignKey("scan_findings.id"))
    file          = Column(String)
    original_code = Column(Text)        # exact snippet to be replaced
    suggested_code = Column(Text)       # replacement snippet
    explanation   = Column(Text)
    diff          = Column(Text)        # unified-diff preview
    status        = Column(String, default="ready")  # ready|unapplicable|committed
    commit_sha    = Column(String)
    created_at    = Column(DateTime, default=datetime.datetime.utcnow)


class GitHubSession(Base):
    """Server-side session holding a user's GitHub OAuth token.

    The primary key is the random, unguessable value stored in the HttpOnly
    session cookie; the access token never reaches the browser.
    """
    __tablename__ = "github_sessions"

    id           = Column(String, primary_key=True)   # = cookie value (random)
    login        = Column(String)
    avatar_url   = Column(String)
    access_token = Column(String)
    created_at   = Column(DateTime, default=datetime.datetime.utcnow)



DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./code_review_bot.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    """Create all database tables if they do not yet exist, then run migrations.

    Safe to call on every application startup — SQLAlchemy's ``create_all`` is
    idempotent for existing tables, and :func:`_run_lightweight_migrations`
    handles schema drift for columns added after a database was first created.
    """
    Base.metadata.create_all(engine)
    _run_lightweight_migrations()


def _run_lightweight_migrations() -> None:
    """Add columns that were introduced after a DB was first created.

    SQLAlchemy's create_all() only creates missing tables, not missing
    columns, so older databases need the new columns added via ALTER TABLE.
    """
    inspector = inspect(engine)
    if "scans" not in inspector.get_table_names():
        return
    existing = {col["name"] for col in inspector.get_columns("scans")}
    if "file_list" not in existing:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE scans ADD COLUMN file_list TEXT"))
