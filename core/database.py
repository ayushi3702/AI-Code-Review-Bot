from __future__ import annotations
import uuid
import datetime
import os
from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()


class AgentRun(Base):
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
    report_markdown = Column(Text)
    report_html   = Column(Text)
    error         = Column(Text)
    created_at    = Column(DateTime, default=datetime.datetime.utcnow)
    finished_at   = Column(DateTime)


class ScanFindingRow(Base):
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


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./code_review_bot.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    Base.metadata.create_all(engine)
