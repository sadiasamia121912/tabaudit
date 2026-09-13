"""Run tabaudit on well-known public datasets and record what it finds.

Usage (from the repo root, with the venv active):

    python benchmarks/run_benchmarks.py              # all datasets
    python benchmarks/run_benchmarks.py titanic adult # just these

Datasets come from OpenML via scikit-learn and are cached in benchmarks/openml_cache/
(git-ignored; re-downloaded automatically if missing). Results go to
benchmarks/results.json, which docs/benchmarks.md is written from.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table
from sklearn.datasets import fetch_openml

from tabaudit import Severity, run_audit

# ---------------------------------------------------------------------------
# 1. What to audit.  (openml name, openml version, target column)
#    Version pins matter: OpenML hosts several copies of "adult" etc. and they differ.
# ---------------------------------------------------------------------------
DATASETS: list[tuple[str, int, str]] = [
    ("titanic", 1, "survived"),
    ("adult", 2, "class"),
    ("credit-g", 1, "class"),
    ("telco-customer-churn", 1, "Churn"),
    ("creditcard", 1, "Class"),
]

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "openml_cache"
RESULTS_PATH = HERE / "results.json"

# "Headline" = a finding serious enough to count the dataset as having a real problem.
HEADLINE = {Severity.CRITICAL, Severity.HIGH}

console = Console()


# ---------------------------------------------------------------------------
# 2. Audit one dataset and boil the report down to a JSON-friendly dict.
# ---------------------------------------------------------------------------
def audit_one(name: str, version: int, target: str) -> dict:
    t0 = time.perf_counter()
    bunch = fetch_openml(
        name, version=version, data_home=str(CACHE_DIR), as_frame=True, parser="auto"
    )
    df = bunch.frame  # features + target in one DataFrame, exactly what a user would have
    report = run_audit(df, target=target)  # max_rows left at the 50 000 default on purpose

    findings = report.sorted_findings()
    return {
        "dataset": name,
        "openml_version": version,
        "target": target,
        "n_rows": len(df),
        "n_cols": int(df.shape[1]),
        "score": report.score,
        "grade": report.grade,
        "counts": {sev.value: report.count(sev) for sev in Severity},
        "has_headline_issue": any(f.severity in HEADLINE for f in findings),
        # Everything a reader needs to judge the finding, minus bulky evidence blobs.
        "findings": [
            {
                "check": f.check,
                "severity": f.severity.value,
                "title": f.title,
                "detail": f.detail,
                "columns": f.columns,
            }
            for f in findings
        ],
        "check_errors": [c.note for c in report.checks if c.status == "error"],
        "seconds": round(time.perf_counter() - t0, 1),
    }


# ---------------------------------------------------------------------------
# 3. Loop over datasets. One failure (e.g. a dropped download) must not kill the run.
# ---------------------------------------------------------------------------
def main(only: list[str]) -> None:
    selected = [d for d in DATASETS if not only or d[0] in only]
    if not selected:
        sys.exit(f"No datasets matched {only}. Known: {[d[0] for d in DATASETS]}")

    results: list[dict] = []
    for name, version, target in selected:
        console.print(f"[cyan]auditing[/] {name} …", end=" ")
        try:
            row = audit_one(name, version, target)
            console.print(f"score {row['score']} ({row['grade']}) in {row['seconds']}s")
        except Exception as exc:  # keep going and report it
            row = {"dataset": name, "error": f"{type(exc).__name__}: {exc}"}
            console.print(f"[red]failed:[/] {row['error']}")
        results.append(row)

    # Merge into any previous run so auditing a subset doesn't wipe the other rows.
    previous = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else []
    merged = {r["dataset"]: r for r in previous}
    merged.update({r["dataset"]: r for r in results})
    results = [merged[name] for name, _, _ in DATASETS if name in merged]  # keep DATASETS order

    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print_summary(results)
    console.print(f"\nwrote {RESULTS_PATH.relative_to(HERE.parent)}")


# ---------------------------------------------------------------------------
# 4. A terminal table so you can eyeball the outcome without opening the JSON.
# ---------------------------------------------------------------------------
def print_summary(results: list[dict]) -> None:
    table = Table(title="tabaudit benchmark", show_lines=False)
    for col in ("dataset", "rows", "score", "grade", "crit", "high", "med", "headline finding"):
        table.add_column(col, justify="right" if col in {"rows", "score"} else "left")

    for r in results:
        if "error" in r:
            table.add_row(r["dataset"], "-", "-", "-", "-", "-", "-", f"[red]{r['error']}[/]")
            continue
        top = next((f for f in r["findings"] if Severity(f["severity"]) in HEADLINE), None)
        c = r["counts"]
        table.add_row(
            r["dataset"],
            f"{r['n_rows']:,}",
            str(r["score"]),
            r["grade"],
            str(c["critical"]),
            str(c["high"]),
            str(c["medium"]),
            top["title"] if top else "[green]none[/]",
        )
    console.print(table)

    ok = [r for r in results if "error" not in r]
    n_hit = sum(r["has_headline_issue"] for r in ok)
    console.print(f"\n[bold]{n_hit} of {len(ok)}[/] datasets have a CRITICAL or HIGH finding.")


if __name__ == "__main__":
    main(sys.argv[1:])
