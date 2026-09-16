"""Standalone HTML report (single file, no external assets)."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from tabaudit.findings import AuditReport, Severity

_GRADE_COLOR = {"A": "#3ddc84", "B": "#8ee06e", "C": "#f5c518", "D": "#ff8c1a", "F": "#ff4d4f"}


def render_html(report: AuditReport, out_path: str | Path) -> Path:
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent),
        autoescape=select_autoescape(["html"]),
    )
    env.filters["basename"] = lambda p: Path(str(p)).name
    tpl = env.get_template("template.html")
    html = tpl.render(
        summary=report.summary,
        findings=report.sorted_findings(),
        checks=report.checks,
        score=report.score,
        grade=report.grade,
        verdict=report.verdict,
        score_breakdown=report.score_breakdown,
        grade_color=_GRADE_COLOR[report.grade],
        counts=[(s.value, report.count(s)) for s in Severity],
        generated_at=report.generated_at,
        version=report.version,
    )
    out = Path(out_path)
    out.write_text(html, encoding="utf-8")
    return out
