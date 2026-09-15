"""Measure how well each check finds faults we planted ourselves.

Usage (from the repo root, with the venv active):

    python benchmarks/evaluate.py                       # all datasets, 3 seeds, 4 faults
    python benchmarks/evaluate.py titanic diabetes      # a subset
    python benchmarks/evaluate.py --seeds 1 --faults duplicates leak_copy

For every dataset in run_benchmarks.DATASETS:
  1. build a *clean base*: drop the leaks we already know about, drop exact duplicates;
  2. run the audit once on the base (the "baseline": what it flags before we touch anything);
  3. for each seed and each fault type, plant the fault with benchmarks/inject.py, run the
     relevant checks, and compare what was flagged with what was planted.

Results go to benchmarks/eval_results.json (merged by dataset/seed/fault, so partial runs
are fine) and a summary table is printed. docs/evaluation.md is written from the JSON.

Reading the numbers:
  precision  of the rows we flagged, the share that were really planted
  recall     of the rows we planted, the share we flagged
  FPR        of the innocent original columns, the share the leakage check accused
Planted label flips are uniform-random, which real label noise is not, so recall here is an
upper bound; and a dataset's *pre-existing* noise makes raw precision a lower bound - hence
`precision_excl_baseline`, which ignores rows the tool already flagged before injection.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.datasets import fetch_openml

from inject import (
    INJECTORS,
    Injection,
    inject_duplicates,
    inject_label_flips,
    inject_leak_copy,
    inject_leak_missingness,
)
from run_benchmarks import CACHE_DIR, DATASETS, HERE
from tabaudit import Severity, __version__, run_audit
from tabaudit.context import AuditContext
from tabaudit.loader import infer_task

RESULTS_PATH = HERE / "eval_results.json"
MAX_ROWS = 20_000  # label-noise CV is the bottleneck; 50k on creditcard took 218 s
RANDOM_STATE = 42  # tabaudit's own default; must match so sample scoping is exact
DEFAULT_SEEDS = [0, 1, 2]

# Leaks documented for these datasets (see docs/benchmarks.md). Removed from the clean base so
# the only leak present is the one we plant.
KNOWN_LEAKS: dict[str, list[str]] = {
    "titanic": ["boat", "body"],
    "bank-marketing": ["duration"],
}

# fault name -> (injector, checks that can see it). Running only those keeps the whole
# evaluation to minutes; label_noise needs leakage first so leaky columns are excluded.
FAULTS: dict[str, tuple] = {
    "duplicates": (inject_duplicates, ["duplicates"]),
    "label_flips": (inject_label_flips, ["leakage", "label_noise"]),
    "leak_copy": (inject_leak_copy, ["leakage"]),
    "leak_missingness": (inject_leak_missingness, ["leakage"]),
}
assert set(FAULTS) == set(INJECTORS)

# A column counts as "accused of leaking" when it appears in one of these findings.
LEAK_SEVERITIES = {Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM}

console = Console()


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
def load_clean(name: str, version: int, target: str, rename: dict | None = None) -> pd.DataFrame:
    bunch = fetch_openml(
        name, version=version, data_home=str(CACHE_DIR), as_frame=True, parser="auto"
    )
    df = bunch.frame
    if rename:
        df = df.rename(columns=rename)
    df = df.drop(columns=[c for c in KNOWN_LEAKS.get(name, []) if c in df.columns])
    return df.drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Pulling "what got flagged" out of a report
# ---------------------------------------------------------------------------
def leak_flagged(report) -> set[str]:
    return {
        c
        for f in report.findings
        if f.check == "leakage"
        and f.severity in LEAK_SEVERITIES
        and "identifier-like" not in f.title
        for c in f.columns
    }


def noise_rows(report) -> tuple[set[int], set[int]]:
    """(suspected, likely) row labels from the label-noise finding, empty if none."""
    for f in report.findings:
        if f.check == "label_noise":
            return set(f.evidence["rows"]), set(f.evidence["rows_likely"])
    return set(), set()


def dup_rows(report) -> set[int]:
    for f in report.findings:
        if f.check == "duplicates" and "rows" in f.evidence:
            return set(f.evidence["rows"])
    return set()


def prf(pred: set, truth: set) -> dict:
    tp = len(pred & truth)
    return {
        "n_flagged": len(pred),
        "n_planted": len(truth),
        "tp": tp,
        "precision": round(tp / len(pred), 4) if pred else None,
        "recall": round(tp / len(truth), 4) if truth else None,
    }


# ---------------------------------------------------------------------------
# 3. Scoring one (dataset, seed, fault)
# ---------------------------------------------------------------------------
def score_injection(inj: Injection, target: str, original_cols: list[str], baseline: dict) -> dict:
    checks = FAULTS[inj.kind][1]
    t0 = time.perf_counter()
    report = run_audit(inj.df, target=target, checks=checks, max_rows=MAX_ROWS)
    out: dict = {"seconds": round(time.perf_counter() - t0, 1)}
    errors = [c.note for c in report.checks if c.status == "error"]
    if errors:
        out["check_errors"] = errors

    if inj.kind == "duplicates":
        out.update(prf(dup_rows(report), set(inj.rows)))

    elif inj.kind == "label_flips":
        # Label noise runs on a sample when the frame is big; only planted flips inside that
        # sample can possibly be found, so recall is measured against those.
        ctx = AuditContext(
            df=inj.df,
            target=target,
            task=infer_task(inj.df[target]),
            max_rows=MAX_ROWS,
            random_state=RANDOM_STATE,
        )
        truth = set(inj.rows) & set(ctx.sample_index())
        suspected, likely = noise_rows(report)
        base = baseline["noise_rows"]
        for tier, pred in (("suspected", suspected), ("likely", likely)):
            m = prf(pred, truth)
            fresh = pred - base  # rows the tool did NOT already flag before we injected
            m["precision_excl_baseline"] = (
                round(len(fresh & truth) / len(fresh), 4) if fresh else None
            )
            out[tier] = m
        out["n_planted_total"] = len(inj.rows)
        out["n_in_sample"] = len(truth)

    else:  # leaks
        flagged = leak_flagged(report)
        planted = set(inj.columns)
        innocent = [c for c in original_cols if c != target]
        false_pos = sorted(flagged & set(innocent))
        severity = next(
            (
                f.severity.value
                for f in report.sorted_findings()
                if f.check == "leakage" and planted & set(f.columns)
            ),
            None,
        )
        out.update(
            {
                "detected": planted <= flagged,
                "severity": severity,
                "false_positive_columns": false_pos,
                "fpr": round(len(false_pos) / len(innocent), 4) if innocent else None,
                "n_innocent_columns": len(innocent),
            }
        )
    return out


def evaluate_dataset(
    name: str, df: pd.DataFrame, target: str, seeds: list[int], faults: list[str]
) -> list[dict]:
    """Baseline once, then every seed x fault. Returns one result dict per run."""
    base_report = run_audit(df, target=target, checks=["leakage", "label_noise"], max_rows=MAX_ROWS)
    baseline = {
        "leak_columns": sorted(leak_flagged(base_report)),
        "noise_rows": noise_rows(base_report)[0],
    }
    console.print(
        f"  base: {len(df):,} rows x {df.shape[1]} cols; already flags "
        f"{len(baseline['noise_rows']):,} noisy rows, leaks {baseline['leak_columns'] or 'none'}"
    )
    results = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for fault in faults:
            inj = FAULTS[fault][0](df, target, rng)
            row = {
                "dataset": name,
                "seed": seed,
                "fault": fault,
                "n_rows": len(df),
                "baseline_leak_columns": baseline["leak_columns"],
                "baseline_n_noise_rows": len(baseline["noise_rows"]),
            }
            try:
                row.update(score_injection(inj, target, list(df.columns), baseline))
            except Exception as exc:  # one failure must not kill the run
                row["error"] = f"{type(exc).__name__}: {exc}"
            results.append(row)
            console.print(f"  seed {seed} {fault:17s} {_one_line(row)}")
    return results


def _one_line(r: dict) -> str:
    if "error" in r:
        return f"[red]error: {r['error']}[/]"
    if r["fault"] == "duplicates":
        return f"recall {r['recall']}  precision {r['precision']}  ({r['seconds']}s)"
    if r["fault"] == "label_flips":
        s, k = r["suspected"], r["likely"]
        return (
            f"suspected P {s['precision']} R {s['recall']} | likely P {k['precision']} "
            f"R {k['recall']} | P excl. baseline {s['precision_excl_baseline']}  ({r['seconds']}s)"
        )
    mark = "[green]detected[/]" if r["detected"] else "[red]MISSED[/]"
    fps = r["false_positive_columns"] or ""
    return f"{mark} as {r['severity']}  FPR {r['fpr']} {fps}  ({r['seconds']}s)"


# ---------------------------------------------------------------------------
# 4. Aggregation + table
# ---------------------------------------------------------------------------
def _mean_sd(values: list) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "-"
    return f"{np.mean(vals):.2f} +/- {np.std(vals):.2f}" if len(vals) > 1 else f"{vals[0]:.2f}"


def print_summary(results: list[dict]) -> None:
    ok = [r for r in results if "error" not in r]
    table = Table(title=f"tabaudit fault-injection evaluation ({len(ok)} runs)")
    for col in ("check", "fault", "metric", "mean +/- sd", "runs"):
        table.add_column(col)

    dup = [r for r in ok if r["fault"] == "duplicates"]
    if dup:
        table.add_row(
            "duplicates", "5% copies", "recall", _mean_sd([r["recall"] for r in dup]), str(len(dup))
        )
        table.add_row("", "", "precision", _mean_sd([r["precision"] for r in dup]), "")

    flips = [r for r in ok if r["fault"] == "label_flips"]
    for tier in ("suspected", "likely"):
        rows = [r[tier] for r in flips]
        if rows:
            table.add_row(
                "label_noise",
                f"3% flips / {tier}",
                "precision",
                _mean_sd([m["precision"] for m in rows]),
                str(len(rows)),
            )
            table.add_row(
                "",
                "",
                "precision excl. baseline",
                _mean_sd([m["precision_excl_baseline"] for m in rows]),
                "",
            )
            table.add_row("", "", "recall", _mean_sd([m["recall"] for m in rows]), "")

    for fault in ("leak_copy", "leak_missingness"):
        rows = [r for r in ok if r["fault"] == fault]
        if rows:
            hits = sum(r["detected"] for r in rows)
            table.add_row("leakage", fault, "detected", f"{hits}/{len(rows)}", str(len(rows)))
            table.add_row("", "", "FPR", _mean_sd([r["fpr"] for r in rows]), "")
    console.print(table)
    errs = [r for r in results if "error" in r]
    if errs:
        console.print(f"[red]{len(errs)} run(s) errored[/]; see eval_results.json")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("datasets", nargs="*", help="subset of run_benchmarks.DATASETS names")
    ap.add_argument(
        "--seeds", type=int, default=len(DEFAULT_SEEDS), help="number of seeds (0..n-1)"
    )
    ap.add_argument("--faults", nargs="*", choices=list(FAULTS), default=list(FAULTS))
    args = ap.parse_args()

    selected = [d for d in DATASETS if not args.datasets or d[0] in args.datasets]
    if not selected:
        raise SystemExit(f"No datasets matched {args.datasets}. Known: {[d[0] for d in DATASETS]}")
    seeds = list(range(args.seeds))

    results: list[dict] = []
    for name, version, target, *extra in selected:
        console.print(f"[cyan]{name}[/]")
        try:
            df = load_clean(name, version, target, *extra)
        except Exception as exc:
            console.print(f"  [red]load failed:[/] {type(exc).__name__}: {exc}")
            continue
        results.extend(evaluate_dataset(name, df, target, seeds, args.faults))

    previous = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {}
    runs = {(r["dataset"], r["seed"], r["fault"]): r for r in previous.get("runs", [])}
    runs.update({(r["dataset"], r["seed"], r["fault"]): r for r in results})
    order = {d[0]: i for i, d in enumerate(DATASETS)}
    merged = sorted(
        runs.values(), key=lambda r: (order.get(r["dataset"], 99), r["seed"], r["fault"])
    )
    payload = {
        "tabaudit_version": __version__,
        "max_rows": MAX_ROWS,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": merged,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_summary(merged)
    console.print(f"\nwrote {RESULTS_PATH.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
