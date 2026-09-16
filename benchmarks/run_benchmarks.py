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
from tabaudit.fix import apply_fixes

# ---------------------------------------------------------------------------
# 1. What to audit.  (openml name, openml version, target column)
#    Version pins matter: OpenML hosts several copies of "adult" etc. and they differ.
# ---------------------------------------------------------------------------
# Optional 4th element: column renames. OpenML anonymises some datasets (V1, V2, ...);
# we restore the names documented by the original source so findings are readable.
BANK_MARKETING_COLS = [
    "age", "job", "marital", "education", "default", "balance", "housing", "loan",
    "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome",
]  # fmt: skip
DATASETS: list[tuple] = [
    ("titanic", 1, "survived"),
    ("adult", 2, "class"),
    ("credit-g", 1, "class"),
    ("telco-customer-churn", 1, "Churn"),
    ("bank-marketing", 1, "Class", {f"V{i}": c for i, c in enumerate(BANK_MARKETING_COLS, 1)}),
    ("breast-w", 1, "Class"),
    ("heart-statlog", 1, "class"),
    ("diabetes", 1, "class"),
    ("spambase", 1, "class"),
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
def audit_one(name: str, version: int, target: str, rename: dict | None = None) -> dict:
    t0 = time.perf_counter()
    bunch = fetch_openml(
        name, version=version, data_home=str(CACHE_DIR), as_frame=True, parser="auto"
    )
    df = bunch.frame  # features + target in one DataFrame, exactly what a user would have
    if rename:
        df = df.rename(columns=rename)
    report = run_audit(df, target=target)  # max_rows left at the 50 000 default on purpose

    # What `tabaudit fix` does unasked: only the fixes with exactly one right answer, no flags.
    t_fix = time.perf_counter()
    clean, plan = apply_fixes(df, report, target=target)
    touched = any(s.n_rows or s.n_columns for s in plan.applied)
    after = run_audit(clean, target=target) if touched else report
    fix_seconds = round(time.perf_counter() - t_fix, 1)

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
        "fix": {
            "rows_removed": plan.rows_before - plan.rows_after,
            "cols_removed": plan.cols_before - plan.cols_after,
            "n_applied": len(plan.applied),
            "n_skipped": len(plan.skipped),
            "flags_offered": plan.flags_offered,
            "applied": [f"{s.check}: {s.note}" for s in plan.applied],
            "score_after": after.score,
            "grade_after": after.grade,
            "seconds": fix_seconds,
        },
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
    for name, version, target, *extra in selected:
        console.print(f"[cyan]auditing[/] {name} …", end=" ")
        try:
            row = audit_one(name, version, target, *extra)
            console.print(f"score {row['score']} ({row['grade']}) in {row['seconds']}s")
        except Exception as exc:  # keep going and report it
            row = {"dataset": name, "error": f"{type(exc).__name__}: {exc}"}
            console.print(f"[red]failed:[/] {row['error']}")
        results.append(row)

    # Merge into any previous run so auditing a subset doesn't wipe the other rows.
    previous = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else []
    merged = {r["dataset"]: r for r in previous}
    merged.update({r["dataset"]: r for r in results})
    results = [merged[d[0]] for d in DATASETS if d[0] in merged]  # keep DATASETS order

    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print_summary(results)
    console.print(f"\nwrote {RESULTS_PATH.relative_to(HERE.parent)}")


# ---------------------------------------------------------------------------
# 4. A terminal table so you can eyeball the outcome without opening the JSON.
# ---------------------------------------------------------------------------
def print_summary(results: list[dict]) -> None:
    table = Table(title="tabaudit benchmark", show_lines=False)
    cols = ("dataset", "rows", "score", "after", "safe fixes", "crit", "high", "med", "headline")
    for col in cols:
        table.add_column(col, justify="right" if col in {"rows", "score", "after"} else "left")

    for r in results:
        if "error" in r:
            table.add_row(r["dataset"], "-", "-", "-", "-", "-", "-", f"[red]{r['error']}[/]")
            continue
        top = next((f for f in r["findings"] if Severity(f["severity"]) in HEADLINE), None)
        c, fx = r["counts"], r.get("fix", {})
        removed = []
        if fx.get("rows_removed"):
            removed.append(f"-{fx['rows_removed']:,} rows")
        if fx.get("cols_removed"):
            removed.append(f"-{fx['cols_removed']} col")
        after = fx.get("score_after", r["score"])
        style = "red" if after < r["score"] else "green" if after > r["score"] else "grey50"
        table.add_row(
            r["dataset"],
            f"{r['n_rows']:,}",
            f"{r['score']} {r['grade']}",
            f"[{style}]{after} {fx.get('grade_after', r['grade'])}[/]",
            ", ".join(removed) or "[grey50]none[/]",
            str(c["critical"]),
            str(c["high"]),
            str(c["medium"]),
            top["title"] if top else "[green]none[/]",
        )
    console.print(table)

    ok = [r for r in results if "error" not in r]
    n_hit = sum(r["has_headline_issue"] for r in ok)
    console.print(f"\n[bold]{n_hit} of {len(ok)}[/] datasets have a CRITICAL or HIGH finding.")

    # The sanity check that matters: a safe fix must never make a dataset score worse.
    worse = [r for r in ok if r.get("fix", {}).get("score_after", r["score"]) < r["score"]]
    if worse:
        console.print("[bold red]REGRESSION[/] - safe fixes lowered the score on:")
        for r in worse:
            console.print(f"  {r['dataset']}: {r['score']} -> {r['fix']['score_after']}")
    else:
        console.print("[green]Safe fixes never lowered a score.[/]")


if __name__ == "__main__":
    main(sys.argv[1:])
