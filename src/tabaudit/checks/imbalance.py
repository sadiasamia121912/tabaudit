"""Class imbalance and rare classes."""

from __future__ import annotations

from tabaudit.context import AuditContext
from tabaudit.findings import Finding, Severity

CHECK = "imbalance"


def run(ctx: AuditContext) -> list[Finding]:
    if ctx.task != "classification" or ctx.y is None:
        return []
    y = ctx.y.dropna()
    counts = y.value_counts()
    if len(counts) < 2:
        return [
            Finding(
                check=CHECK,
                severity=Severity.CRITICAL,
                title="Target has only one class",
                detail=f"All {len(y):,} labelled rows are '{counts.index[0]}'.",
                recommendation="This is not a classification dataset as-is.",
                columns=[ctx.target or ""],
            )
        ]
    dist = {str(k): int(v) for k, v in counts.items()}
    minority_frac = float(counts.min() / counts.sum())
    ratio = float(counts.max() / counts.min())
    findings: list[Finding] = []

    sev: Severity | None
    if ratio >= 100:
        sev = Severity.HIGH
    elif ratio >= 10:
        sev = Severity.MEDIUM
    elif ratio >= 3:
        sev = Severity.LOW
    else:
        sev = None
    if sev is not None:
        findings.append(
            Finding(
                check=CHECK,
                severity=sev,
                title=f"Class imbalance {ratio:,.0f}:1 (minority class = {minority_frac:.2%})",
                detail="Accuracy is meaningless here - a model predicting the majority class "
                f"scores {1 - minority_frac:.1%} without learning anything.",
                recommendation="Report precision/recall, PR-AUC or balanced accuracy; use "
                "stratified splits and class weights. Never oversample before splitting.",
                columns=[ctx.target or ""],
                evidence={"class_counts": dist, "imbalance_ratio": round(ratio, 2)},
            )
        )

    rare = counts[counts < 10]
    if not rare.empty:
        findings.append(
            Finding(
                check=CHECK,
                severity=Severity.MEDIUM,
                title=f"{len(rare)} class(es) have fewer than 10 examples",
                detail=", ".join(f"'{k}' ({v})" for k, v in rare.items()),
                recommendation="Merge into an 'other' class or collect more data; "
                "stratified CV will fail or be meaningless for these.",
                columns=[ctx.target or ""],
                evidence={"rare_classes": {str(k): int(v) for k, v in rare.items()}},
            )
        )
    return findings
