"""Rich terminal rendering of an AuditReport."""

from __future__ import annotations

from rich import box
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tabaudit.findings import AuditReport, Finding, Severity

SEV_STYLE = {
    Severity.CRITICAL: ("bold white on red", "CRITICAL"),
    Severity.HIGH: ("bold white on dark_orange", "HIGH"),
    Severity.MEDIUM: ("bold black on yellow", "MEDIUM"),
    Severity.LOW: ("bold black on bright_cyan", "LOW"),
    Severity.INFO: ("bold white on grey42", "INFO"),
}
GRADE_COLOR = {"A": "green", "B": "green3", "C": "yellow", "D": "dark_orange", "F": "red"}


def _badge(sev: Severity) -> Text:
    style, label = SEV_STYLE[sev]
    return Text(f" {label} ", style=style)


def _score_bar(score: int, width: int = 30) -> Text:
    filled = round(width * score / 100)
    color = "green" if score >= 75 else "yellow" if score >= 50 else "red"
    bar = Text()
    bar.append("█" * filled, style=color)
    bar.append("░" * (width - filled), style="grey37")
    return bar


def _summary_table(report: AuditReport) -> Table:
    s = report.summary
    t = Table.grid(padding=(0, 3))
    t.add_column(style="bold grey70")
    t.add_column()
    t.add_row("Dataset", s.path)
    t.add_row("Shape", f"{s.n_rows:,} rows × {s.n_cols} columns  ({s.memory_mb} MB)")
    kinds = ", ".join(f"{v} {k}" for k, v in s.dtypes.items())
    t.add_row("Columns", kinds)
    if s.target:
        cls = f"  ({s.n_classes} classes)" if s.n_classes else ""
        t.add_row("Target", f"{s.target}  →  {s.task}{cls}")
    else:
        t.add_row("Target", "[dim]none (unsupervised checks only)[/dim]")
    if s.test_path:
        t.add_row("Test set", f"{s.test_path}  ({s.n_test_rows:,} rows)")
    return t


def _score_panel(report: AuditReport) -> Panel:
    color = GRADE_COLOR[report.grade]
    grade = Text(f" {report.grade} ", style=f"bold white on {color}")
    line1 = Text.assemble(
        "Data Health Score  ", (f"{report.score}", f"bold {color}"), ("/100  ", "grey70"), grade
    )
    line2 = _score_bar(report.score)
    line3 = Text(report.verdict, style=f"italic {color}")
    counts = Text()
    for sev in Severity:
        n = report.count(sev)
        if n:
            counts.append_text(_badge(sev))
            counts.append(f" {n}   ")
    return Panel(
        Group(line1, line2, Text(), line3, Text(), counts),
        title="[bold]Verdict[/bold]",
        border_style=color,
        padding=(1, 2),
    )


def _finding_block(f: Finding) -> Panel:
    head = Text.assemble(_badge(f.severity), "  ", (f.title, "bold"))
    body = Text()
    body.append(f.detail, style="grey85")
    if f.recommendation:
        body.append("\n\n→ ", style="bold green")
        body.append(f.recommendation, style="green")
    top = f.evidence.get("top_suspects") if f.evidence else None
    parts: list = [body]
    if top:
        tbl = Table(
            box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold grey70"
        )
        tbl.add_column("row", justify="right")
        tbl.add_column("given")
        tbl.add_column("suggested")
        tbl.add_column("P(given)", justify="right")
        tbl.add_column("P(suggested)", justify="right")
        for r in top[:10]:
            tbl.add_row(
                str(r["row"]),
                str(r["given_label"]),
                f"[bold]{r['suggested_label']}[/bold]",
                f"{r['confidence_in_given']:.2f}",
                f"{r['confidence_in_suggested']:.2f}",
            )
        parts.append(Padding(tbl, (1, 0, 0, 0)))
    return Panel(
        Group(*parts),
        title=head,
        title_align="left",
        subtitle=f"[dim]{f.check}[/dim]",
        subtitle_align="right",
        border_style=SEV_STYLE[f.severity][0].split(" on ")[-1],
        padding=(0, 2),
    )


def _checks_table(report: AuditReport) -> Table:
    t = Table(box=box.SIMPLE, header_style="bold grey70", show_edge=False)
    t.add_column("check")
    t.add_column("status")
    t.add_column("findings", justify="right")
    t.add_column("time", justify="right")
    for c in report.checks:
        status = {
            "ok": "[green]✓ ok[/green]",
            "error": f"[red]✗ error[/red] [dim]{c.note}[/dim]",
        }.get(c.status, c.status)
        t.add_row(c.name, status, str(c.n_findings), f"{c.duration_s:.2f}s")
    return t


def render_console(
    report: AuditReport, console: Console | None = None, verbose: bool = True
) -> None:
    con = console or Console()
    con.print()
    con.print(
        Panel(
            _summary_table(report),
            title=f"[bold cyan]tabaudit[/bold cyan] [dim]v{report.version}[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    con.print(_score_panel(report))
    findings = report.sorted_findings()
    if not findings:
        con.print(Panel("[bold green]No issues found.[/bold green]", border_style="green"))
    else:
        con.print(f"[bold]Findings[/bold] [dim]({len(findings)})[/dim]")
        for f in findings:
            con.print(_finding_block(f))
    if verbose:
        con.print(_checks_table(report))
    con.print()
