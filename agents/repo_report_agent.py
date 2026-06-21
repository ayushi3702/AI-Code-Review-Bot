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
import logging

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
    seen: set[tuple] = set()
    out: list[ScanFinding] = []
    for f in findings:
        key = (f.file, f.line, f.title.lower()[:50])
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def _score(findings: list[ScanFinding]) -> tuple[int, str]:
    penalty = sum(SEVERITY_WEIGHT.get(f.severity, 3) for f in findings)
    score = max(0, 100 - penalty)
    grade = (
        "A" if score >= 90 else
        "B" if score >= 80 else
        "C" if score >= 70 else
        "D" if score >= 60 else "F"
    )
    return score, grade


def _counts(findings: list[ScanFinding]) -> dict:
    c = {"high": 0, "medium": 0, "low": 0}
    for f in findings:
        c[f.severity] = c.get(f.severity, 0) + 1
    return c


# ── Markdown ─────────────────────────────────────────────────────────────────

def build_markdown(state: ScanState, findings: list[ScanFinding], score: int, grade: str) -> str:
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
    lines.append(f"*Generated {datetime.datetime.utcnow():%Y-%m-%d %H:%M UTC} by the AI Code Review Platform "
                 "(security · performance · architecture · quality agents, run in parallel).*")
    return "\n".join(lines)


# ── HTML ─────────────────────────────────────────────────────────────────────

_HTML_TEMPLATE = Template("""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Code Review — {{ repo }}</title>
<style>
  :root { --hi:#e5484d; --med:#f5a623; --lo:#3b82f6; --bg:#0d1117; --card:#161b22; --fg:#e6edf3; --muted:#8b949e; }
  * { box-sizing:border-box; } body { margin:0; font-family:system-ui,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--fg); }
  .wrap { max-width:960px; margin:0 auto; padding:32px 20px 80px; }
  h1 { font-size:1.6rem; margin:0 0 4px; } .sub { color:var(--muted); margin:0 0 24px; }
  .score { display:flex; align-items:center; gap:20px; background:var(--card); border:1px solid #30363d; border-radius:14px; padding:20px 24px; margin-bottom:24px; }
  .grade { font-size:3rem; font-weight:800; width:84px; height:84px; display:flex; align-items:center; justify-content:center; border-radius:50%; }
  .gA{color:#3fb950;border:3px solid #3fb950} .gB{color:#7ee787;border:3px solid #7ee787} .gC{color:var(--med);border:3px solid var(--med)} .gD{color:#db6d28;border:3px solid #db6d28} .gF{color:var(--hi);border:3px solid var(--hi)}
  .pills span { display:inline-block; padding:4px 10px; border-radius:999px; font-size:.8rem; margin-right:8px; }
  .p-high{background:rgba(229,72,77,.15);color:var(--hi)} .p-med{background:rgba(245,166,35,.15);color:var(--med)} .p-low{background:rgba(59,130,246,.15);color:var(--lo)}
  .agent { margin-top:28px; } .agent h2 { font-size:1.15rem; border-bottom:1px solid #30363d; padding-bottom:8px; }
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
    <section class="agent">
      <h2>{{ agent_label(agent) }} ({{ items|length }})</h2>
      {% for f in items %}
      <div class="finding {{ f.severity }}">
        <div class="title">{{ emoji(f.severity) }} {{ f.title }}</div>
        <div class="loc">{{ f.file }}{% if f.line %}:{{ f.line }}{% endif %}</div>
        {% if f.detail %}<p class="detail">{{ f.detail }}</p>{% endif %}
        {% if f.recommendation %}<div class="fix">💡 {{ f.recommendation }}</div>{% endif %}
      </div>
      {% endfor %}
    </section>
    {% endfor %}
  {% endif %}
  <footer>Generated {{ ts }} UTC · AI Code Review Platform — security · performance · architecture · quality.</footer>
</div></body></html>""")


def build_html(state: ScanState, findings: list[ScanFinding], score: int, grade: str) -> str:
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
        ts=f"{datetime.datetime.utcnow():%Y-%m-%d %H:%M}",
    )


# ── Agent entry point ─────────────────────────────────────────────────────────

async def repo_report_agent(state: ScanState) -> dict:
    db = SessionLocal()
    try:
        findings = _dedupe(state.findings)
        score, grade = _score(findings)
        markdown = build_markdown(state, findings, score, grade)
        html_doc = build_html(state, findings, score, grade)

        scan = db.query(Scan).filter(Scan.id == state.scan_id).first()
        if scan:
            scan.status = "done"
            scan.stage = "done"
            scan.finding_count = len(findings)
            scan.score = score
            scan.grade = grade
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

        logger.info("[report] score=%d grade=%s findings=%d", score, grade, len(findings))
        return {
            "report_markdown": markdown,
            "report_html": html_doc,
            "score": score,
            "grade": grade,
            "completed_agents": ["report"],
        }
    except Exception as e:
        logger.error("[report] %s", e)
        return {"errors": [f"report: {e}"]}
    finally:
        db.close()
