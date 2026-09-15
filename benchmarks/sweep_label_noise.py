"""Which flagging rule, and which threshold, best recovers planted label flips?

Usage (repo root, venv active):

    python benchmarks/sweep_label_noise.py              # all datasets, 3 seeds
    python benchmarks/sweep_label_noise.py diabetes heart-statlog --seeds 1

The label-noise check has two stages: an out-of-fold model that produces class
probabilities for every row (expensive), and a *rule* that turns those probabilities into
"this row is suspect" (cheap). This script computes the probabilities once per
(dataset, seed) and scores every rule x threshold combination on the same matrix, against
the flips planted by benchmarks/inject.py. `self_conf` below is the probability the model
gives the row's *given* label.

Rules compared:
  cl                 cleanlab confident_learning filter               (v0.1.0 "suspected")
  cl & sc<t          confident_learning AND self_conf < t             (v0.1.0 "likely", t=0.2)
  sc<t               self_conf < t alone                              (chosen: 0.2 likely, 0.3 suspected)
  argmax!=y & sc<t   model's top class differs from the label AND self_conf < t

Scored exactly like evaluate.py: precision *excluding rows the same rule already flagged on
the clean base* (the dataset's own noise), recall on planted flips inside the sample, F1 of
those two, and the share of all rows the rule flags (a rule that flags 15% of every dataset
is useless whatever its F1). Writes benchmarks/sweep_results.json and prints a table.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from evaluate import MAX_ROWS, load_clean
from inject import inject_label_flips
from run_benchmarks import DATASETS, HERE
from tabaudit.checks import label_noise, leakage
from tabaudit.context import AuditContext
from tabaudit.loader import infer_task

try:  # cleanlab is no longer a dependency; install it to reproduce the `cl` rows
    from cleanlab.filter import find_label_issues
except ImportError:  # pragma: no cover
    find_label_issues = None

RESULTS_PATH = HERE / "sweep_results.json"
THRESHOLDS = [0.1, 0.2, 0.3, 0.4]
FLIP_FRAC = 0.03
console = Console()


# ---------------------------------------------------------------------------
# 1. Probabilities once, rules many times
# ---------------------------------------------------------------------------
def oof(df: pd.DataFrame, target: str) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    """(row index, y codes, out-of-fold probs) the way the audit itself would compute them:
    leakage runs first so leaky / ID columns are excluded from the model's inputs."""
    ctx = AuditContext(df=df, target=target, task=infer_task(df[target]), max_rows=MAX_ROWS)
    leakage.run(ctx)
    out = label_noise.out_of_fold_probs(ctx)
    if out is None:
        raise RuntimeError("label-noise check would skip this dataset")
    index, y_codes, _classes, probs = out
    return index, y_codes, probs


def rule_masks(y_codes: np.ndarray, probs: np.ndarray) -> dict[str, np.ndarray]:
    """Every rule as a boolean mask over the rows of `probs`."""
    n = len(y_codes)
    self_conf = probs[np.arange(n), y_codes]
    disagree = probs.argmax(axis=1) != y_codes
    masks: dict[str, np.ndarray] = {}
    if find_label_issues is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cl = find_label_issues(
                labels=y_codes, pred_probs=probs, filter_by="confident_learning", n_jobs=1
            )
        masks["cl"] = cl
        for t in THRESHOLDS:
            masks[f"cl & sc<{t}"] = cl & (self_conf < t)
    for t in THRESHOLDS:
        masks[f"sc<{t}"] = self_conf < t
        masks[f"argmax!=y & sc<{t}"] = disagree & (self_conf < t)
    return masks


# ---------------------------------------------------------------------------
# 2. Score one (dataset, seed)
# ---------------------------------------------------------------------------
def score_dataset(name: str, df: pd.DataFrame, target: str, seeds: list[int]) -> list[dict]:
    base_index, base_y, base_probs = oof(df, target)
    base_flags = {rule: set(base_index[m]) for rule, m in rule_masks(base_y, base_probs).items()}
    rows = []
    for seed in seeds:
        inj = inject_label_flips(df, target, np.random.default_rng(seed), frac=FLIP_FRAC)
        index, y_codes, probs = oof(inj.df, target)
        truth = set(inj.rows) & set(index)
        for rule, mask in rule_masks(y_codes, probs).items():
            pred = set(index[mask])
            fresh = pred - base_flags[rule]
            tp, tp_fresh = len(pred & truth), len(fresh & truth)
            p_excl = tp_fresh / len(fresh) if fresh else 0.0
            r = tp / len(truth) if truth else 0.0
            rows.append(
                {
                    "dataset": name,
                    "seed": seed,
                    "rule": rule,
                    "n_rows": len(index),
                    "n_planted": len(truth),
                    "n_flagged": len(pred),
                    "flagged_frac": round(len(pred) / len(index), 4),
                    "precision": round(tp / len(pred), 4) if pred else None,
                    "precision_excl_baseline": round(p_excl, 4),
                    "recall": round(r, 4),
                    "f1": round(2 * p_excl * r / (p_excl + r), 4) if p_excl + r else 0.0,
                }
            )
        console.print(f"  seed {seed}: {len(truth)} flips in sample, {len(index):,} rows")
    return rows


# ---------------------------------------------------------------------------
# 3. Aggregate and print
# ---------------------------------------------------------------------------
def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for rule in dict.fromkeys(r["rule"] for r in rows):
        rs = [r for r in rows if r["rule"] == rule]
        out.append(
            {
                "rule": rule,
                "runs": len(rs),
                "precision": round(float(np.mean([r["precision"] or 0 for r in rs])), 3),
                "precision_excl_baseline": round(
                    float(np.mean([r["precision_excl_baseline"] for r in rs])), 3
                ),
                "recall": round(float(np.mean([r["recall"] for r in rs])), 3),
                "f1": round(float(np.mean([r["f1"] for r in rs])), 3),
                "flagged_frac": round(float(np.mean([r["flagged_frac"] for r in rs])), 4),
            }
        )
    return sorted(out, key=lambda r: -r["f1"])


def print_summary(summary: list[dict]) -> None:
    table = Table(title="label-noise rule sweep (mean over datasets x seeds, 3% planted flips)")
    for col in ("rule", "P raw", "P excl. base", "recall", "F1", "flagged %", "runs"):
        table.add_column(col, justify="right" if col != "rule" else "left")
    for r in summary:
        mark = " <- v0.1.0 'likely'" if r["rule"] == "cl & sc<0.2" else ""
        table.add_row(
            r["rule"] + mark,
            f"{r['precision']:.2f}",
            f"{r['precision_excl_baseline']:.2f}",
            f"{r['recall']:.2f}",
            f"{r['f1']:.2f}",
            f"{100 * r['flagged_frac']:.1f}",
            str(r["runs"]),
        )
    console.print(table)


def markdown_table(summary: list[dict]) -> str:
    lines = [
        "| rule | precision (raw) | precision excl. baseline | recall | F1 | rows flagged |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| `{r['rule']}` | {r['precision']:.2f} | {r['precision_excl_baseline']:.2f} | "
            f"{r['recall']:.2f} | **{r['f1']:.2f}** | {100 * r['flagged_frac']:.1f} % |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("datasets", nargs="*")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    selected = [d for d in DATASETS if not args.datasets or d[0] in args.datasets]
    if not selected:
        raise SystemExit(f"No datasets matched {args.datasets}")

    rows: list[dict] = []
    for name, version, target, *extra in selected:
        console.print(f"[cyan]{name}[/]")
        try:
            df = load_clean(name, version, target, *extra)
            rows.extend(score_dataset(name, df, target, list(range(args.seeds))))
        except Exception as exc:
            console.print(f"  [red]failed:[/] {type(exc).__name__}: {exc}")

    summary = summarise(rows)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "thresholds": THRESHOLDS,
                "flip_frac": FLIP_FRAC,
                "summary": summary,
                "runs": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print_summary(summary)
    console.print("\nMarkdown for docs/checks.md:\n")
    print(markdown_table(summary))
    console.print(f"\nwrote {RESULTS_PATH.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
