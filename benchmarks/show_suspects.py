"""Show the top label-noise suspects of a benchmark dataset so a human can judge them.

Usage (from the repo root, with the venv active):

    python benchmarks/show_suspects.py heart-statlog          # top 5 suspects
    python benchmarks/show_suspects.py credit-g --top 8
    python benchmarks/show_suspects.py diabetes --out docs/label_noise_review.md

For each suspect it prints one table: every feature, the suspect's value, and what a
"typical" row of each class looks like (median for numbers, most common value for
categories). The question to answer for each row is simply: *does this row look more
like the class it was given, or the class the model suggests?*

The point of this script is roadmap task 1.5 - the tool only ranks suspects; whether
they are real errors is a judgement call that has to be made by looking at the rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.datasets import fetch_openml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_benchmarks import CACHE_DIR, DATASETS  # sibling script, not a package
from tabaudit import run_audit

console = Console()


def load(name: str) -> tuple[pd.DataFrame, str]:
    spec = next((d for d in DATASETS if d[0] == name), None)
    if spec is None:
        sys.exit(f"Unknown dataset {name!r}. Known: {[d[0] for d in DATASETS]}")
    _, version, target, *extra = spec
    bunch = fetch_openml(
        name, version=version, data_home=str(CACHE_DIR), as_frame=True, parser="auto"
    )
    df = bunch.frame
    if extra:
        df = df.rename(columns=extra[0])
    return df, target


def typical(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """One column per class: median for numeric features, mode for everything else."""
    feats = [c for c in df.columns if c != target]
    out = {}
    for label, grp in df.groupby(target, observed=True):
        col = {}
        for c in feats:
            s = grp[c].dropna()
            if s.empty:
                col[c] = "-"
            elif pd.api.types.is_numeric_dtype(s):
                col[c] = f"{s.median():g}"
            else:
                col[c] = str(s.mode().iloc[0])
        out[str(label)] = col
    return pd.DataFrame(out)


def fmt(v) -> str:
    if pd.isna(v):
        return "[dim]NaN[/]"
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--top", type=int, default=5, help="how many suspects to show")
    ap.add_argument("--out", type=Path, help="also append a markdown review sheet here")
    args = ap.parse_args()

    df, target = load(args.dataset)
    console.print(f"[cyan]auditing[/] {args.dataset} ({len(df):,} rows) ...")
    report = run_audit(df, target=target)
    finding = next((f for f in report.findings if f.check == "label_noise"), None)
    if finding is None:
        sys.exit("No label-noise finding on this dataset.")

    ev = finding.evidence
    console.print(
        f"[bold]{finding.title}[/]  (showing top {args.top} of {len(ev['top_suspects'])} listed)\n"
    )
    typ = typical(df, target)
    lines = [
        f"\n## {args.dataset} - label-noise review\n",
        "| row | given | suggested | verdict | why |",
        "|---|---|---|---|---|",
    ]

    for s in ev["top_suspects"][: args.top]:
        row = df.loc[s["row"]]
        given, sugg = s["given_label"], s["suggested_label"]
        t = Table(
            title=f"row {s['row']}: given [red]{given}[/] (model: {s['confidence_in_given']:.0%}) "
            f"-> suggested [green]{sugg}[/] ({s['confidence_in_suggested']:.0%})",
            show_lines=False,
        )
        t.add_column("feature")
        t.add_column("this row", style="bold")
        t.add_column(f"typical {given}")
        t.add_column(f"typical {sugg}")
        for c in typ.index:
            t.add_row(str(c), fmt(row[c]), typ.at[c, given], typ.at[c, sugg])
        console.print(t)
        lines.append(f"| {s['row']} | {given} | {sugg} |  |  |")

    console.print(
        "\n[dim]For each row, ask: does it look more like the 'typical' column for its given "
        "label, or for the suggested one? Fill in the verdict column of the review sheet.[/]"
    )
    if args.out:
        with args.out.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        console.print(f"appended review sheet to {args.out}")


if __name__ == "__main__":
    main()
