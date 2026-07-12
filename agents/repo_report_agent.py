"""Report agent: turn raw findings into a rated, categorised report.

Outputs both formats the spec calls for:
  * Markdown — great for CLI output, GitHub, and copy-paste.
  * HTML — a standalone styled page for the demo / the React app to embed.

It also computes a 0-100 health score and an A-F grade by weighting findings by
severity, so a viewer gets a one-glance verdict before reading details.
"""
from __future__ import annotations
import datetime
import html
import json
import logging

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

from jinja2 import Template

from core.state import ScanState, ScanFinding
from core.database import SessionLocal, Scan, ScanFindingRow

logger = logging.getLogger(__name__)

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
SEVERITY_WEIGHT = {"high": 12, "medium": 5, "low": 1}
SEVERITY_EMOJI = {"high": "🔴", "medium": "🟡", "low": "🔵"}
AGENT_LABEL = {
    "security": "Security",
    "performance": "Performance",
    "architecture": "Architecture",
    "quality": "Code Quality",
}


def _dedupe(findings: list[ScanFinding]) -> list[ScanFinding]:
    """Remove duplicate findings from the merged agent output.

    Since four agents run concurrently and can each query overlapping code
    regions, the same issue may be reported more than once (same file, line,
    and title).  Duplicates are identified by the ``(file, line, title[:50])``
    triple and only the first occurrence is kept.

    Args:
        findings: Raw merged list from all four agents.

    Returns:
        De-duplicated list preserving the original ordering.
    """
    seen: set[tuple] = set()
    out: list[ScanFinding] = []
    for f in findings:
        key = (f.file, f.line, f.title.lower()[:50])
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def _score(findings: list[ScanFinding], file_count: int = 0) -> tuple[int, str]:
    """Compute a 0–100 health score and an A–F grade for the scanned repository.

    Rather than penalising a repo purely for the raw count of findings (which
    would unfairly punish larger codebases), the score is based on the
    *weighted density* of issues per file::

        density = sum(SEVERITY_WEIGHT[f.severity] for f in findings) / file_count
        penalty = min(100, density * 6)   # 6 pts per weighted-issue-per-file
        score   = max(0, round(100 - penalty))

    Grade thresholds: A ≥ 90, B ≥ 80, C ≥ 70, D ≥ 60, F < 60.

    Args:
        findings:   De-duplicated list of findings.
        file_count: Total number of source files in the repo (used for density
                    normalisation; clamped to 1 to avoid division by zero).

    Returns:
        A ``(score, grade)`` tuple, e.g. ``(87, 'B')``.
    """
    # Normalize by repo size so large repos aren't punished just for being big:
    # we score the *density* of weighted issues per file rather than a raw sum.
    weighted = sum(SEVERITY_WEIGHT.get(f.severity, 3) for f in findings)
    files = max(file_count, 1)
    density = weighted / files               # weighted issues per file
    penalty = min(100, density * 6)          # 6 pts per weighted-issue-per-file, capped
    score = max(0, round(100 - penalty))
    grade = (
        "A" if score >= 90 else
        "B" if score >= 80 else
        "C" if score >= 70 else
        "D" if score >= 60 else "F"
    )
    return score, grade


def _counts(findings: list[ScanFinding]) -> dict:
    """Count findings by severity level.

    Args:
        findings: List of :class:`~core.state.ScanFinding` objects.

    Returns:
        A dict ``{'high': n, 'medium': n, 'low': n}``.
    """
    c = {"high": 0, "medium": 0, "low": 0}
    for f in findings:
        c[f.severity] = c.get(f.severity, 0) + 1
    return c


# ── Markdown ─────────────────────────────────────────────────────────────────

def build_markdown(state: ScanState, findings: list[ScanFinding], score: int, grade: str) -> str:
    """Render the scan results as a Markdown report string.

    Produces a human-readable report suitable for display in the CLI, pasting
    into a GitHub comment, or saving to disk.  Findings are grouped by agent
    and sorted by severity within each group.

    Args:
        state:    Completed :class:`~core.state.ScanState` for metadata
                  (repo name, file count, languages, etc.).
        findings: De-duplicated, sorted list of findings.
        score:    Computed health score (0–100).
        grade:    Letter grade (A–F).

    Returns:
        A Markdown string starting with a heading and health summary.
    """
    counts = _counts(findings)
    lines = [
        f"# 🔍 Code Review Report — `{state.repo_name}`",
        "",
        f"**Health score: {score}/100 (Grade {grade})**  ",
        f"Scanned **{state.file_count} files** · **{state.chunk_count} code chunks** · "
        f"languages: {', '.join(f'{k} ({v})' for k, v in sorted(state.languages.items()))}",
        "",
        f"**{counts['high']} high · {counts['medium']} medium · {counts['low']} low** severity findings",
        "",
    ]
    if not findings:
        lines.append("✅ No significant issues found across the repository. Nice work!")
        return "\n".join(lines)

    by_agent: dict[str, list[ScanFinding]] = {}
    for f in sorted(findings, key=lambda x: SEVERITY_ORDER.get(x.severity, 3)):
        by_agent.setdefault(f.agent, []).append(f)

    for agent, items in by_agent.items():
        lines.append(f"## {AGENT_LABEL.get(agent, agent.title())} ({len(items)})")
        for f in items:
            emoji = SEVERITY_EMOJI.get(f.severity, "⚪")
            loc = f"`{f.file}`" + (f":{f.line}" if f.line else "")
            lines.append(f"- {emoji} **{f.title}** — {loc}")
            if f.detail:
                lines.append(f"  - {f.detail}")
            if f.recommendation:
                lines.append(f"  - 💡 _Fix:_ {f.recommendation}")
        lines.append("")

    lines.append("---")
    lines.append(f"*Generated {datetime.datetime.now(IST):%Y-%m-%d %H:%M IST} by the AI Code Review Platform "
                 "(security · performance · architecture · quality agents, run in parallel).*")
    return "\n".join(lines)


# ── HTML ─────────────────────────────────────────────────────────────────────

_HTML_TEMPLATE = Template("""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Code Review — {{ repo }}</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E%F0%9F%94%8D%3C/text%3E%3C/svg%3E"/>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap" rel="stylesheet"/>
<style>
  :root { --hi:#e5484d; --med:#f5a623; --lo:#3b82f6; --bg:#0d1117; --card:#161b22; --fg:#e6edf3; --muted:#8b949e; }
  * { box-sizing:border-box; } body { margin:0; font-family:"Poppins",system-ui,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--fg); }
  .wrap { max-width:960px; margin:0 auto; padding:32px 20px 80px; }
  h1 { font-size:1.6rem; margin:0 0 4px; } .sub { color:var(--muted); margin:0 0 24px; }
  .score { display:flex; align-items:center; gap:20px; background:var(--card); border:1px solid #30363d; border-radius:14px; padding:20px 24px; margin-bottom:24px; }
  .grade { font-size:3rem; font-weight:800; width:84px; height:84px; display:flex; align-items:center; justify-content:center; border-radius:50%; }
  .gA{color:#3fb950;border:3px solid #3fb950} .gB{color:#7ee787;border:3px solid #7ee787} .gC{color:var(--med);border:3px solid var(--med)} .gD{color:#db6d28;border:3px solid #db6d28} .gF{color:var(--hi);border:3px solid var(--hi)}
  .pills span { display:inline-block; padding:4px 10px; border-radius:999px; font-size:.8rem; margin-right:8px; }
  .p-high{background:rgba(229,72,77,.15);color:var(--hi)} .p-med{background:rgba(245,166,35,.15);color:var(--med)} .p-low{background:rgba(59,130,246,.15);color:var(--lo)}
  .agent { margin-top:28px; } .agent h2 { font-size:1.15rem; border-bottom:1px solid #30363d; padding-bottom:8px; }
  .agent > summary { cursor:pointer; list-style:none; user-select:none; display:flex; align-items:center; gap:8px; }
  .agent > summary::-webkit-details-marker { display:none; }
  .agent > summary::before { content:"\u25be"; color:var(--muted); font-size:.9rem; transition:transform .15s ease; }
  .agent:not([open]) > summary::before { transform:rotate(-90deg); }
  .agent > summary h2 { margin:0; flex:1; }
  .finding { background:var(--card); border:1px solid #30363d; border-left:4px solid var(--muted); border-radius:10px; padding:14px 16px; margin:12px 0; }
  .finding.high{border-left-color:var(--hi)} .finding.medium{border-left-color:var(--med)} .finding.low{border-left-color:var(--lo)}
  .finding .title { font-weight:600; } .finding .loc { color:var(--muted); font-size:.85rem; font-family:ui-monospace,monospace; }
  .finding .detail { margin:8px 0 0; color:#c9d1d9; } .finding .fix { margin-top:8px; color:#7ee787; font-size:.9rem; }
  footer { margin-top:40px; color:var(--muted); font-size:.8rem; }
  .ok { background:var(--card); border:1px solid #30363d; border-radius:12px; padding:24px; color:#3fb950; }
</style></head><body><div class="wrap">
  <h1>🔍 {{ repo }}</h1>
  <p class="sub">{{ file_count }} files · {{ chunk_count }} chunks · {{ langs }}</p>
  <div class="score">
    <div class="grade g{{ grade }}">{{ grade }}</div>
    <div>
      <div style="font-size:1.4rem;font-weight:700;">{{ score }}/100 health</div>
      <div class="pills" style="margin-top:8px;">
        <span class="p-high">{{ counts.high }} high</span>
        <span class="p-med">{{ counts.medium }} medium</span>
        <span class="p-low">{{ counts.low }} low</span>
      </div>
    </div>
  </div>
  {% if not findings %}
    <div class="ok">✅ No significant issues found across the repository.</div>
  {% else %}
    {% for agent, items in by_agent.items() %}
    <details class="agent" open>
      <summary><h2>{{ agent_label(agent) }} ({{ items|length }})</h2></summary>
      {% for f in items %}
      <div class="finding {{ f.severity }}">
        <div class="title">{{ emoji(f.severity) }} {{ f.title }}</div>
        <div class="loc">{{ f.file }}{% if f.line %}:{{ f.line }}{% endif %}</div>
        {% if f.detail %}<p class="detail">{{ f.detail }}</p>{% endif %}
        {% if f.recommendation %}<div class="fix">💡 {{ f.recommendation }}</div>{% endif %}
      </div>
      {% endfor %}
    </details>
    {% endfor %}
  {% endif %}
  <footer>Generated {{ ts }} IST · AI Code Review Platform — security · performance · architecture · quality.</footer>
</div></body></html>""")


def build_html(state: ScanState, findings: list[ScanFinding], score: int, grade: str) -> str:
    """Render the scan results as a standalone, self-contained HTML report.

    Uses a Jinja2 template with embedded CSS to produce a dark-themed page that
    works as a static file (no external dependencies except Google Fonts).  The
    report groups findings by agent in collapsible ``<details>`` sections and
    colour-codes them by severity.

    Args:
        state:    Completed :class:`~core.state.ScanState` for metadata.
        findings: De-duplicated, sorted list of findings.
        score:    Computed health score (0–100).
        grade:    Letter grade (A–F).

    Returns:
        A complete HTML document string ready to be written to a ``.html`` file
        or served directly from the ``/api/scan/{id}/report.html`` endpoint.
    """
    by_agent: dict[str, list[ScanFinding]] = {}
    for f in sorted(findings, key=lambda x: SEVERITY_ORDER.get(x.severity, 3)):
        by_agent.setdefault(f.agent, []).append(f)

    return _HTML_TEMPLATE.render(
        repo=html.escape(state.repo_name),
        file_count=state.file_count,
        chunk_count=state.chunk_count,
        langs=", ".join(f"{k} ({v})" for k, v in sorted(state.languages.items())),
        grade=grade,
        score=score,
        counts=_counts(findings),
        findings=findings,
        by_agent=by_agent,
        agent_label=lambda a: AGENT_LABEL.get(a, a.title()),
        emoji=lambda s: SEVERITY_EMOJI.get(s, "⚪"),
        ts=f"{datetime.datetime.now(IST):%Y-%m-%d %H:%M}",
    )


# ── Agent entry point ─────────────────────────────────────────────────────────

async def repo_report_agent(state: ScanState) -> dict:
    """LangGraph node: compile findings into reports and persist the final scan record.

    This node is the fan-in point of the pipeline — it only executes once all
    four specialist agents have completed.  Its responsibilities are:

    1. De-duplicate findings from all four agents.
    2. Compute the health score and letter grade.
    3. Render Markdown and HTML reports.
    4. Persist the final state (findings, score, grade, reports) to the database.

    Args:
        state: :class:`~core.state.ScanState` with the merged findings from all
               four parallel agents.

    Returns:
        Dict containing ``report_markdown``, ``report_html``, ``score``,
        ``grade``, and ``completed_agents`` keys for LangGraph.
    """
    db = SessionLocal()
    try:
        logger.info(
            "report agent started — scan_id=%s raw_findings=%d",
            state.scan_id, len(state.findings),
        )
        findings = _dedupe(state.findings)
        score, grade = _score(findings, state.file_count)
        markdown = build_markdown(state, findings, score, grade)
        html_doc = build_html(state, findings, score, grade)

        scan = db.query(Scan).filter(Scan.id == state.scan_id).first()
        if scan:
            scan.status = "done"
            scan.stage = "done"
            scan.finding_count = len(findings)
            scan.score = score
            scan.grade = grade
            scan.file_list = json.dumps(state.file_index or [])
            scan.report_markdown = markdown
            scan.report_html = html_doc
            scan.finished_at = datetime.datetime.utcnow()
            for f in findings:
                db.add(ScanFindingRow(
                    scan_id=state.scan_id, agent=f.agent, severity=f.severity,
                    file=f.file, line=f.line, title=f.title, detail=f.detail,
                    recommendation=f.recommendation, code_snippet=f.code_snippet,
                ))
            db.commit()
        else:
            logger.warning("report agent: scan row not found in DB — scan_id=%s", state.scan_id)

        logger.info(
            "report agent completed — scan_id=%s score=%d grade=%s findings=%d",
            state.scan_id, score, grade, len(findings),
        )
        return {
            "report_markdown": markdown,
            "report_html": html_doc,
            "score": score,
            "grade": grade,
            "completed_agents": ["report"],
        }
    except Exception as e:
        logger.error(
            "report agent failed — scan_id=%s: %s",
            state.scan_id, e, exc_info=True,
        )
        return {"errors": [f"report: {e}"]}
    finally:
        db.close()
