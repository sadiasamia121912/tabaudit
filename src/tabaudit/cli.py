"""Command-line interface: `tabaudit audit`, `fix`, `gate`, `demo`, `checks`."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.status import Status
from rich.table import Table

from tabaudit import __version__
from tabaudit.audit import run_audit
from tabaudit.checks import REGISTRY
from tabaudit.findings import Severity
from tabaudit.fix import FixPlan, apply_fixes
from tabaudit.loader import load_table, write_table
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


def _parse_fail_on(value: str | None) -> Severity | None:
    if value is None or value.lower() == "none":
        return None
    try:
        return Severity(value.lower())
    except ValueError:
        choices = [s.value for s in Severity if s != Severity.INFO] + ["none"]
        err.print(f"[bold red]error:[/bold red] --fail-on must be one of {choices}")
        raise typer.Exit(code=2) from None


def _gate_failures(report, fail_under: int | None, fail_sev: Severity | None) -> list[str]:
    """Why this report fails the CI gate - empty list means it passes."""
    reasons = []
    if fail_under is not None and report.score < fail_under:
        reasons.append(f"score {report.score} < {fail_under}")
    if fail_sev is not None:
        # rank: CRITICAL=0 ... INFO=4, so "this severity or worse" is rank <= threshold rank
        hits = [f for f in report.findings if f.severity.rank <= fail_sev.rank]
        if hits:
            worst = min(hits, key=lambda f: f.severity.rank)
            reasons.append(
                f"{len(hits)} finding(s) at {fail_sev.value} or worse "
                f"(worst: {worst.severity.value} - {worst.title})"
            )
    return reasons


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
    fail_sev = _parse_fail_on(fail_on)
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

    reasons = _gate_failures(report, fail_under, fail_sev)
    for reason in reasons:
        err.print(f"[bold red]FAIL[/bold red] {reason}")
    if reasons:
        raise typer.Exit(code=1)


def _plan_table(plan: FixPlan) -> Table:
    t = Table(box=box.SIMPLE, header_style="bold grey70", show_edge=False)
    t.add_column("")
    t.add_column("finding")
    t.add_column("action")
    t.add_column("result")
    for s in plan.steps:
        mark = "[green]✔[/green]" if s.applied else "[yellow]?[/yellow]"
        what = f"[bold]{s.action}[/bold]"
        if s.columns:
            shown = ", ".join(s.columns[:4])
            if len(s.columns) > 4:
                shown += f" +{len(s.columns) - 4}"
            what += f"\n[grey62]{shown}[/grey62]"
        note = s.note if s.applied else f"[yellow]{s.note}[/yellow]"
        t.add_row(mark, f"{s.title}\n[dim]{s.check}[/dim]", what, note)
    return t


@app.command()
def fix(
    data: Path = typer.Argument(..., help="CSV / TSV / Parquet / Feather / JSON-lines file."),
    target: str | None = typer.Option(None, "--target", "-t", help="Label column name."),
    test: Path | None = typer.Option(
        None, "--test", help="Held-out set, so rows leaked into it can be dropped from train."
    ),
    drop_leaky: bool = typer.Option(
        False,
        "--drop-leaky",
        help="Also drop the columns flagged as leaky or identifier-like. "
        "Only you know when a column becomes available, so this is off by default.",
    ),
    flag_noise: bool = typer.Option(
        False,
        "--flag-noise",
        help="Add a boolean `tabaudit_suspect` column marking likely-mislabeled rows. "
        "Never relabels or drops them.",
    ),
    out: Path | None = typer.Option(
        None, "--out", "-o", help="Where to write the cleaned data (default: <name>.clean.csv)."
    ),
    plan_out: Path | None = typer.Option(
        None, "--plan", help="Where to write the fix plan JSON (default: <name>.fixplan.json)."
    ),
    checks: str | None = typer.Option(
        None, "--checks", "-c", help="Comma-separated subset of checks (see `tabaudit checks`)."
    ),
    max_rows: int = typer.Option(
        50_000, help="Row cap for model-based checks (sampled above this)."
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only print the summary line."),
) -> None:
    """Audit a dataset, then apply only the fixes that have exactly one right answer.

    Duplicates, constant columns, leftover index columns, numbers stored as text and rows with
    no label are fixed outright. Suspected leaks and label errors wait for [bold]--drop-leaky[/bold]
    / [bold]--flag-noise[/bold]; skipping them is the correct outcome, not a failure, so the
    exit code stays 0. Scaling, encoding and imputing are never written to the file - they
    belong in a pipeline fitted on the training fold.
    """
    selected = [c.strip() for c in checks.split(",")] if checks else None
    try:
        with Status("[cyan]auditing…", console=console, spinner="dots") as status:

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
        df = load_table(data)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        err.print(f"[bold red]error:[/bold red] {exc}")
        raise typer.Exit(code=2) from None

    flags = [f for f, on in (("--drop-leaky", drop_leaky), ("--flag-noise", flag_noise)) if on]
    clean, plan = apply_fixes(df, report, flags, target=target)

    out_path = Path(out) if out else data.with_name(f"{data.stem}.clean{data.suffix or '.csv'}")
    plan_path = Path(plan_out) if plan_out else out_path.with_name(f"{data.stem}.fixplan.json")
    try:
        write_table(clean, out_path)
    except ValueError as exc:
        err.print(f"[bold red]error:[/bold red] {exc}")
        raise typer.Exit(code=2) from None
    plan_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")

    unchanged = plan.rows_after == plan.rows_before and plan.cols_after == plan.cols_before
    if unchanged and not plan.applied:
        shape = (
            f"unchanged: {plan.rows_after:,} rows, {plan.cols_after} columns "
            "— nothing here has exactly one right answer"
        )
    else:
        shape = (
            f"{plan.rows_before:,} → {plan.rows_after:,} rows, "
            f"{plan.cols_before} → {plan.cols_after} columns"
        )
    if quiet:
        console.print(
            f"tabaudit fix: {len(plan.applied)} applied, {len(plan.skipped)} skipped; {shape}"
        )
    else:
        console.print()
        console.print(f"[bold]Health score before fixing[/bold]  {report.score}/100 {report.grade}")
        console.print(_plan_table(plan))
        console.print(f"[bold]{shape}[/bold]")
        if plan.flags_offered:
            console.print(
                f"[yellow]{len(plan.skipped)} fix(es) need your say-so:[/yellow] "
                f"{' '.join(plan.flags_offered)}  [dim](read the findings first: tabaudit audit)[/dim]"
            )
    console.print(f"[dim]cleaned data →[/dim] {out_path.resolve()}")
    console.print(f"[dim]fix plan →[/dim] {plan_path.resolve()}")
    if not quiet:
        tgt = f" --target {target}" if target else ""
        console.print(f"[dim]check it:[/dim] tabaudit audit {out_path}{tgt}")


@app.command()
def gate(
    files: list[Path] = typer.Argument(..., help="One or more data files."),
    target: str | None = typer.Option(
        None, "--target", "-t", help="Label column name (must be the same in every file)."
    ),
    checks: str | None = typer.Option(
        None, "--checks", "-c", help="Comma-separated subset of checks (see `tabaudit checks`)."
    ),
    max_rows: int = typer.Option(
        50_000, help="Row cap for model-based checks (sampled above this)."
    ),
    fail_under: int | None = typer.Option(
        None, "--fail-under", help="Fail a file whose health score is below this."
    ),
    fail_on: str | None = typer.Option(
        "high",
        "--fail-on",
        help="Fail a file with any finding of this severity or worse "
        "(critical | high | medium | low), or 'none' to gate on the score only. Default: high.",
    ),
) -> None:
    """Audit several files and print one pass/fail line each (for pre-commit and CI).

    Exit code 1 if any file fails the gate, 2 if a file cannot be audited.
    """
    selected = [c.strip() for c in checks.split(",")] if checks else None
    fail_sev = _parse_fail_on(fail_on)
    failed = errored = 0
    for path in files:
        try:
            report = run_audit(path, target=target, checks=selected, max_rows=max_rows)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            console.print(f"[bold red]✖[/bold red] {path}: {exc}")
            errored += 1
            continue
        reasons = _gate_failures(report, fail_under, fail_sev)
        line = f"{path}: score {report.score}/100 (grade {report.grade})"
        if reasons:
            console.print(f"[bold red]✖[/bold red] {line} - " + "; ".join(reasons))
            failed += 1
        else:
            console.print(f"[bold green]✔[/bold green] {line}")
    if errored:
        raise typer.Exit(code=2)
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
