"""Duplicate rows, conflicting labels, and train/test contamination."""

from __future__ import annotations

import pandas as pd

from tabaudit.context import AuditContext
from tabaudit.findings import Finding, Severity

CHECK = "duplicates"


def _row_hashes(df: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(df, index=False)


def run(ctx: AuditContext) -> list[Finding]:
    df = ctx.df
    findings: list[Finding] = []
    n = len(df)
    feats = ctx.feature_cols

    # ---- exact duplicate rows (all columns) -----------------------------
    full_dup = df.duplicated(keep="first")
    n_full = int(full_dup.sum())
    if n_full:
        frac = n_full / n
        sev = Severity.HIGH if frac >= 0.05 else Severity.MEDIUM if frac >= 0.01 else Severity.LOW
        findings.append(
            Finding(
                check=CHECK,
                severity=sev,
                title=f"{n_full:,} exact duplicate rows ({frac:.1%})",
                detail="Identical rows will land in both train and validation folds, "
                "inflating every cross-validation metric.",
                recommendation="`df.drop_duplicates()` before splitting.",
                evidence={"n_duplicates": n_full, "fraction": round(frac, 4)},
            )
        )

    # ---- same features, different label --------------------------------
    if ctx.target and feats:
        fh = _row_hashes(df[feats])
        grp = df.groupby(fh.values)[ctx.target].nunique(dropna=False)
        conflicting = grp[grp > 1]
        if not conflicting.empty:
            n_rows = int(fh.isin(conflicting.index).sum())
            findings.append(
                Finding(
                    check=CHECK,
                    severity=Severity.MEDIUM,
                    title=f"{len(conflicting):,} feature-identical group(s) carry conflicting labels",
                    detail=f"{n_rows:,} rows share identical features with another row but have a "
                    "different target value. No model can fit these; they cap achievable accuracy.",
                    recommendation="Investigate the labelling process, or add the feature that "
                    "actually distinguishes them.",
                    evidence={"n_groups": len(conflicting), "n_rows": n_rows},
                )
            )

    # ---- train/test overlap --------------------------------------------
    if ctx.test_df is not None:
        common = [c for c in feats if c in ctx.test_df.columns]
        if common:
            train_h = set(_row_hashes(df[common]))
            test_h = _row_hashes(ctx.test_df[common])
            n_leak = int(test_h.isin(train_h).sum())
            if n_leak:
                frac = n_leak / len(ctx.test_df)
                sev = Severity.CRITICAL if frac >= 0.01 else Severity.HIGH
                findings.append(
                    Finding(
                        check=CHECK,
                        severity=sev,
                        title=f"{n_leak:,} test rows ({frac:.1%}) also appear in the training set",
                        detail="The model has already seen these rows. Any test-set score is "
                        "partly memorisation, not generalisation.",
                        recommendation="Remove overlapping rows from the test set, or re-split "
                        "with a group-aware splitter if rows belong to entities.",
                        evidence={"n_overlap": n_leak, "fraction_of_test": round(frac, 4)},
                    )
                )
    return findings
