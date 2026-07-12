"""CLI for the repo-wide deep-scan platform.

Usage:
    python cli.py scan <github-repo-url> [--out reports] [--open]

Examples:
    python cli.py scan https://github.com/pallets/flask.git
    python cli.py scan https://github.com/owner/repo.git --out out --open

Triggering is CLI-first today; the same `run_scan()` entry point is reused by
the FastAPI server, so adding a scheduler later is just another caller.
"""
from __future__ import annotations
import os
import sys
import asyncio
import argparse
import datetime
import webbrowser

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agents.repo_orchestrator import run_scan
from core.audit import setup_logging

setup_logging()

console = Console()


def _write_reports(repo_name: str, markdown: str, html_doc: str, out_dir: str) -> tuple[str, str]:
    """Write the Markdown and HTML reports to the output directory.

    Creates the directory if it does not exist, then writes both files with a
    timestamp-based name so successive scans of the same repo do not overwrite
    each other.

    Args:
        repo_name: Short repository name used as the filename prefix.
        markdown:  Markdown report string to write.
        html_doc:  HTML report string to write.
        out_dir:   Directory path to write the files into.

    Returns:
        A ``(md_path, html_path)`` tuple of absolute file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{repo_name}-{stamp}"
    md_path = os.path.join(out_dir, f"{base}.md")
    html_path = os.path.join(out_dir, f"{base}.html")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return md_path, html_path


def _summary_table(state) -> Table:
    """Build a Rich :class:`~rich.table.Table` summarising findings by agent.

    Displays a row per agent (security, performance, architecture, quality)
    with the number of findings each produced, plus a totals row broken down
    by severity (high/medium/low).

    Args:
        state: The completed :class:`~core.state.ScanState` returned by
               :func:`~agents.repo_orchestrator.run_scan`.

    Returns:
        A configured :class:`rich.table.Table` ready to print to the console.
    """
    counts = {"high": 0, "medium": 0, "low": 0}
    for f in state.findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    by_agent: dict[str, int] = {}
    for f in state.findings:
        by_agent[f.agent] = by_agent.get(f.agent, 0) + 1

    table = Table(title="Findings by agent", show_header=True, header_style="bold cyan")
    table.add_column("Agent")
    table.add_column("Findings", justify="right")
    for agent in ("security", "performance", "architecture", "quality"):
        table.add_row(agent.title(), str(by_agent.get(agent, 0)))
    table.add_row("[bold]Total[/bold]",
                  f"[bold]{len(state.findings)}[/bold] "
                  f"([red]{counts['high']}H[/red]/[yellow]{counts['medium']}M[/yellow]/[blue]{counts['low']}L[/blue])")
    return table


async def _run(args: argparse.Namespace) -> int:
    """Execute a single scan and print the results to the terminal.

    Runs :func:`~agents.repo_orchestrator.run_scan` with a Rich spinner, then
    prints the health grade, summary table, and report file paths.  Optionally
    opens the HTML report in the system browser.

    Args:
        args: Parsed :class:`argparse.Namespace` with ``source``, ``out``, and
              ``open`` attributes.

    Returns:
        Exit code: ``0`` on success with no high-severity findings, ``1`` if the
        scan itself failed, ``2`` if any high-severity findings were found (for
        use as a CI gate).
    """
    console.print(Panel.fit(f"[bold]AI Code Review Platform[/bold]\nScanning: [cyan]{args.source}[/cyan]"))

    with console.status("[bold green]Working…[/bold green]") as status:
        def progress(msg: str):
            status.update(f"[bold green]{msg}[/bold green]")
            console.log(msg)
        state = await run_scan(args.source, progress=progress)

    if state.errors and not state.report_markdown:
        console.print(f"[bold red]Scan failed:[/bold red] {'; '.join(state.errors)}")
        return 1

    grade_color = {"A": "green", "B": "green", "C": "yellow", "D": "yellow", "F": "red"}.get(state.grade, "white")
    console.print(Panel.fit(
        f"[bold {grade_color}]Grade {state.grade} — {state.score}/100[/bold {grade_color}]\n"
        f"{state.file_count} files · {state.chunk_count} chunks · "
        f"{', '.join(f'{k}({v})' for k, v in sorted(state.languages.items()))}",
        title="Health",
    ))
    console.print(_summary_table(state))

    md_path, html_path = _write_reports(state.repo_name or "repo", state.report_markdown,
                                        state.report_html, args.out)
    console.print(f"\n[green]✓[/green] Markdown report: [cyan]{md_path}[/cyan]")
    console.print(f"[green]✓[/green] HTML report:     [cyan]{html_path}[/cyan]")

    if args.open:
        webbrowser.open(f"file://{os.path.abspath(html_path)}")

    # non-zero exit if any high-severity issue — lets CI gate on it
    return 2 if any(f.severity == "high" for f in state.findings) else 0


def main() -> None:
    """CLI entry point — parse arguments and run the appropriate sub-command.

    Currently supports a single ``scan`` sub-command::

        python cli.py scan <github-repo-url> [--out DIR] [--open]

    Calls :func:`asyncio.run` to execute the async scan pipeline and forwards
    the integer exit code to :func:`sys.exit`.
    """
    parser = argparse.ArgumentParser(description="AI-powered full-repository code review.")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Scan a GitHub repository.")
    scan.add_argument("source", help="GitHub repository URL.")
    scan.add_argument("--out", default="reports", help="Directory to write reports into.")
    scan.add_argument("--open", action="store_true", help="Open the HTML report when done.")

    args = parser.parse_args()
    if args.command == "scan":
        sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
