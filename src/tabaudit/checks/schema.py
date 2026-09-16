"""Structural problems: missing values, constant columns, numbers stored as text."""

from __future__ import annotations

import pandas as pd

from tabaudit.context import AuditContext
from tabaudit.findings import Finding, Fix, Severity

CHECK = "schema"


def run(ctx: AuditContext) -> list[Finding]:
    df = ctx.df
    findings: list[Finding] = []
    n = len(df)

    # ---- missing values -------------------------------------------------
    miss = df.isna().mean()
    heavy = miss[miss >= 0.5].sort_values(ascending=False)
    if not heavy.empty:
        findings.append(
            Finding(
                check=CHECK,
                severity=Severity.MEDIUM,
                title=f"{len(heavy)} column(s) are >=50% missing",
                detail=", ".join(f"{c} ({v:.0%})" for c, v in heavy.items()),
                recommendation="Drop these columns or justify an imputation strategy; "
                "most models cannot learn from a column that is mostly empty.",
                columns=list(heavy.index),
                evidence={"missing_fraction": {c: round(float(v), 4) for c, v in heavy.items()}},
            )
        )
    if ctx.target and df[ctx.target].isna().any():
        k = int(df[ctx.target].isna().sum())
        findings.append(
            Finding(
                check=CHECK,
                severity=Severity.HIGH,
                title=f"Target '{ctx.target}' has {k} missing value(s)",
                detail=f"{k / n:.2%} of rows have no label.",
                recommendation="Drop unlabeled rows before training or they will crash / bias the model.",
                columns=[ctx.target],
                evidence={"n_missing_target": k, "rows": df.index[df[ctx.target].isna()].tolist()},
                fix=Fix("drop_rows", {"rows": df.index[df[ctx.target].isna()].tolist()}),
            )
        )

    # ---- constant / near-constant columns -------------------------------
    const, near = [], []
    for col in ctx.feature_cols:
        s = df[col]
        nun = s.nunique(dropna=False)
        if nun <= 1:
            const.append(col)
        else:
            top_frac = s.value_counts(dropna=False, normalize=True).iloc[0]
            if top_frac >= 0.99:
                near.append((col, float(top_frac)))
    if const:
        findings.append(
            Finding(
                check=CHECK,
                severity=Severity.LOW,
                title=f"{len(const)} constant column(s)",
                detail=", ".join(const),
                recommendation="Remove - a constant column carries zero information.",
                columns=const,
                fix=Fix("drop_columns", {"columns": const}),
            )
        )
    if near:
        findings.append(
            Finding(
                check=CHECK,
                severity=Severity.LOW,
                title=f"{len(near)} near-constant column(s) (one value >=99%)",
                detail=", ".join(f"{c} ({v:.1%})" for c, v in near),
                recommendation="Usually safe to drop; keep only if the rare value is meaningful.",
                columns=[c for c, _ in near],
                evidence={"top_value_fraction": {c: round(v, 4) for c, v in near}},
            )
        )

    # ---- numbers stored as text ----------------------------------------
    numeric_as_text = []
    for col in ctx.feature_cols:
        s = df[col]
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            sample = s.dropna().astype(str).head(2000)
            if sample.empty:
                continue
            coerced = pd.to_numeric(sample.str.replace(",", "", regex=False), errors="coerce")
            if coerced.notna().mean() >= 0.95:
                numeric_as_text.append(col)
    if numeric_as_text:
        findings.append(
            Finding(
                check=CHECK,
                severity=Severity.LOW,
                title=f"{len(numeric_as_text)} numeric column(s) stored as text",
                detail=", ".join(numeric_as_text),
                recommendation="Cast to numeric; as text they will be one-hot encoded or silently dropped.",
                columns=numeric_as_text,
                fix=Fix("coerce_dtype", {"columns": numeric_as_text, "to": "numeric"}),
            )
        )

    # ---- leftover index columns ----------------------------------------
    junk = [c for c in df.columns if str(c).lower().startswith("unnamed:")]
    if junk:
        findings.append(
            Finding(
                check=CHECK,
                severity=Severity.INFO,
                title="Leftover index column(s) from a previous export",
                detail=", ".join(junk),
                recommendation="Drop them (written by `to_csv` without `index=False`).",
                columns=junk,
                fix=Fix("drop_columns", {"columns": junk}),
            )
        )
    return findings
