"""Command-line interface: `tabaudit audit`, `tabaudit demo`, `tabaudit checks`."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.status import Status

from tabaudit import __version__
from tabaudit.audit import run_audit
from tabaudit.checks import REGISTRY
from tabaudit.findings import Severity
from tabaudit.report import render_console, render_html

app = typer.Typer(
    name="tabaudit",
    help="Audit tabular ML datasets for leakage, duplicates, label noise and imbalance.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def _force_utf8() -> None:
    """Legacy Windows consoles default to cp1252, which cannot encode the glyphs we print."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            with contextlib.suppress(ValueError, AttributeError):
                stream.reconfigure(encoding="utf-8", errors="replace")


_force_utf8()
console = Console()
err = Console(stderr=True)


def _version(value: bool) -> None:
    if value:
        console.print(f"tabaudit {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool | None = typer.Option(
        None, "--version", "-V", callback=_version, is_eager=True, help="Show version and exit."
    ),
) -> None:
    """[bold cyan]tabaudit[/bold cyan] - find the problems in your dataset before your model does."""


@app.command()
def audit(
    data: Path = typer.Argument(..., help="CSV / TSV / Parquet / Feather / JSON-lines file."),
    target: str | None = typer.Option(None, "--target", "-t", help="Label column name."),
    test: Path | None = typer.Option(
        None, "--test", help="Held-out set to check for contamination."
    ),
    html: Path | None = typer.Option(None, "--html", help="Write a standalone HTML report here."),
    json_out: Path | None = typer.Option(None, "--json", help="Write machine-readable JSON here."),
    checks: str | None = typer.Option(
        None, "--checks", "-c", help="Comma-separated subset of checks (see `tabaudit checks`)."
    ),
    max_rows: int = typer.Option(
        50_000, help="Row cap for model-based checks (sampled above this)."
    ),
    fail_under: int | None = typer.Option(
        None, "--fail-under", help="Exit with code 1 if the health score is below this (for CI)."
    ),
    fail_on: str | None = typer.Option(
        None,
        "--fail-on",
        help="Exit with code 1 on any finding of this severity or worse "
        "(critical | high | medium | low), for CI.",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only print the score line."),
) -> None:
    """Audit a dataset and print a health report."""
    selected = [c.strip() for c in checks.split(",")] if checks else None
    fail_sev: Severity | None = None
    if fail_on is not None:
        try:
            fail_sev = Severity(fail_on.lower())
        except ValueError:
            err.print(
                f"[bold red]error:[/bold red] --fail-on must be one of {[s.value for s in Severity]}"
            )
            raise typer.Exit(code=2) from None
    try:
        with Status("[cyan]loading…", console=console, spinner="dots") as status:

            def progress(name: str, state: str) -> None:
                if state == "running":
                    status.update(f"[cyan]running check[/cyan] [bold]{name}[/bold]…")

            report = run_audit(
                data,
                target=target,
                test=test,
                checks=selected,
                max_rows=max_rows,
                on_progress=progress,
            )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        err.print(f"[bold red]error:[/bold red] {exc}")
        raise typer.Exit(code=2) from None

    if quiet:
        console.print(
            f"tabaudit score {report.score}/100 (grade {report.grade}) - {report.verdict}"
        )
    else:
        render_console(report, console)

    if html:
        p = render_html(report, html)
        console.print(f"[dim]HTML report →[/dim] {p.resolve()}")
    if json_out:
        Path(json_out).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        console.print(f"[dim]JSON report →[/dim] {Path(json_out).resolve()}")

    failed = False
    if fail_under is not None and report.score < fail_under:
        err.print(f"[bold red]FAIL[/bold red] score {report.score} < {fail_under}")
        failed = True
    if fail_sev is not None:
        # rank: CRITICAL=0 ... INFO=4, so "this severity or worse" is rank <= threshold rank
        hits = [f for f in report.findings if f.severity.rank <= fail_sev.rank]
        if hits:
            worst = min(hits, key=lambda f: f.severity.rank)
            err.print(
                f"[bold red]FAIL[/bold red] {len(hits)} finding(s) at {fail_sev.value} or worse "
                f"(worst: {worst.severity.value} - {worst.title})"
            )
            failed = True
    if failed:
        raise typer.Exit(code=1)


@app.command()
def demo(
    out_dir: Path = typer.Option(
        Path("tabaudit_demo"), "--out", help="Where to write the demo CSVs."
    ),
    html: bool = typer.Option(True, help="Also write an HTML report into the demo folder."),
) -> None:
    """Generate a synthetic dataset with planted defects and audit it."""
    from tabaudit.demo import write_demo

    train, test = write_demo(out_dir)
    console.print(f"[dim]demo data written to[/dim] {out_dir.resolve()}")
    console.print(
        f"[dim]equivalent command:[/dim] tabaudit audit {train} --target churn --test {test}"
        + (f" --html {out_dir / 'report.html'}" if html else "")
    )
    with Status("[cyan]auditing demo dataset…", console=console, spinner="dots"):
        report = run_audit(train, target="churn", test=test)
    render_console(report, console)
    if html:
        p = render_html(report, out_dir / "report.html")
        console.print(f"[dim]HTML report →[/dim] {p.resolve()}")


@app.command(name="checks")
def list_checks() -> None:
    """List available checks."""
    from rich.table import Table

    t = Table(title="available checks", show_edge=False)
    t.add_column("name", style="bold")
    t.add_column("what it finds")
    for name, fn in REGISTRY.items():
        doc = (fn.__module__ and __import__(fn.__module__, fromlist=["__doc__"]).__doc__) or ""
        t.add_row(name, doc.strip().splitlines()[0] if doc else "")
    console.print(t)


if __name__ == "__main__":  # pragma: no cover
    app()
