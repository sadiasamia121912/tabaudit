"""Target leakage: identifier columns and single features that predict the target too well.

The core idea: a *single* feature that predicts the target almost perfectly is almost
never a real signal - it is a proxy for the answer that was recorded after the fact
(e.g. `churn_reason` filled in only for customers who churned).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from tabaudit.context import AuditContext
from tabaudit.findings import Finding, Severity

CHECK = "leakage"

ID_NAME_HINTS = ("id", "uuid", "guid", "key", "index", "idx", "no", "number", "code", "ref")
MAX_FEATURES_FOR_MODEL = 300
SUSPICIOUS = 0.90  # single-feature AUC / R^2 above this is suspicious
NEAR_PERFECT = 0.98


def _looks_like_id_name(name: str) -> bool:
    low = str(name).lower()
    parts = low.replace("-", "_").split("_")
    return low.endswith("id") or any(p in ID_NAME_HINTS for p in parts)


def _id_like_columns(ctx: AuditContext) -> list[tuple[str, float]]:
    out = []
    n = len(ctx.df)
    if n < 20:
        return out
    for col in ctx.feature_cols:
        s = ctx.df[col]
        if pd.api.types.is_float_dtype(s):
            continue  # continuous features are naturally unique
        uniq = s.nunique(dropna=True) / max(1, s.notna().sum())
        if uniq >= 0.98 or (uniq >= 0.90 and _looks_like_id_name(col)):
            out.append((col, float(uniq)))
    return out


def _fill_sentinel(x: np.ndarray) -> np.ndarray:
    """Replace NaN with a value below the column min so the tree can split on missingness."""
    if not np.isnan(x).any():
        return x
    finite = x[~np.isnan(x)]
    sentinel = (finite.min() - 1.0) if finite.size else -1.0
    return np.where(np.isnan(x), sentinel, x)


def _single_feature_scores(ctx: AuditContext, skip: set[str]) -> dict[str, dict]:
    """Cross-validated predictive power of each feature *alone*."""
    idx = ctx.sample_index()
    Xe = ctx.X_encoded.loc[idx]
    y = ctx.y.loc[idx]
    keep = y.notna().to_numpy()
    Xe, y = Xe[keep], y[keep]
    feats = [c for c in Xe.columns if c not in skip][:MAX_FEATURES_FOR_MODEL]
    results: dict[str, dict] = {}

    if ctx.task == "classification":
        y_codes, classes = pd.factorize(y)
        n_classes = len(classes)
        min_class = np.bincount(y_codes).min()
        n_splits = int(min(5, max(2, min_class)))
        if min_class < 2 or n_classes < 2:
            return results
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=ctx.random_state)
        for col in feats:
            x = _fill_sentinel(Xe[col].to_numpy(dtype=float)).reshape(-1, 1)
            if np.unique(x).size < 2:
                continue
            model = DecisionTreeClassifier(
                max_depth=4, min_samples_leaf=max(5, len(x) // 500), random_state=ctx.random_state
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    proba = cross_val_predict(model, x, y_codes, cv=cv, method="predict_proba")
                    if n_classes == 2:
                        score = roc_auc_score(y_codes, proba[:, 1])
                    else:
                        score = roc_auc_score(y_codes, proba, multi_class="ovr", average="macro")
            except ValueError:
                continue
            results[col] = {"metric": "AUC", "score": float(score)}
    else:
        y_num = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
        ok = ~np.isnan(y_num)
        Xe, y_num = Xe[ok], y_num[ok]
        if len(y_num) < 20:
            return results
        cv = KFold(n_splits=5, shuffle=True, random_state=ctx.random_state)
        for col in feats:
            x = _fill_sentinel(Xe[col].to_numpy(dtype=float)).reshape(-1, 1)
            if np.unique(x).size < 2:
                continue
            model = DecisionTreeRegressor(
                max_depth=6, min_samples_leaf=max(5, len(x) // 500), random_state=ctx.random_state
            )
            try:
                pred = cross_val_predict(model, x, y_num, cv=cv)
                score = r2_score(y_num, pred)
            except ValueError:
                continue
            results[col] = {"metric": "R2", "score": float(score)}
    return results


def _missingness_auc(ctx: AuditContext, col: str) -> float | None:
    """How well does *whether the value is missing* predict the target? (binary only)"""
    if ctx.task != "classification" or ctx.y is None:
        return None
    s = ctx.df[col]
    if not s.isna().any() or s.isna().all():
        return None
    y = ctx.y
    keep = y.notna()
    y_codes, classes = pd.factorize(y[keep])
    if len(classes) != 2:
        return None
    auc = roc_auc_score(y_codes, s[keep].isna().astype(int))
    return float(max(auc, 1 - auc))


def run(ctx: AuditContext) -> list[Finding]:
    findings: list[Finding] = []

    # ---- 1. identifier-like columns ------------------------------------
    ids = _id_like_columns(ctx)
    if ids:
        findings.append(
            Finding(
                check=CHECK,
                severity=Severity.MEDIUM,
                title=f"{len(ids)} identifier-like column(s) present as features",
                detail=", ".join(f"{c} ({u:.0%} unique)" for c, u in ids),
                recommendation="Drop before training. Trees will memorise row identity, and if IDs "
                "are sequential they leak collection order / time.",
                columns=[c for c, _ in ids],
                evidence={"unique_ratio": {c: round(u, 4) for c, u in ids}},
            )
        )
    id_set = {c for c, _ in ids}
    ctx.excluded_features |= id_set

    if ctx.target is None or ctx.task == "unsupervised":
        return findings

    # ---- 2. single-feature predictive power -----------------------------
    scores = _single_feature_scores(ctx, skip=id_set)
    perfect, suspicious = [], []
    for col, r in scores.items():
        if r["score"] >= NEAR_PERFECT:
            perfect.append((col, r))
        elif r["score"] >= SUSPICIOUS:
            suspicious.append((col, r))

    def _describe(col: str, r: dict) -> str:
        txt = f"{col} ({r['metric']}={r['score']:.3f})"
        m_auc = _missingness_auc(ctx, col)
        if m_auc is not None and m_auc >= 0.9:
            txt += f" - its *missingness alone* has AUC {m_auc:.2f}"
        return txt

    if perfect:
        findings.append(
            Finding(
                check=CHECK,
                severity=Severity.CRITICAL,
                title=f"{len(perfect)} feature(s) predict the target almost perfectly on their own",
                detail="; ".join(_describe(c, r) for c, r in perfect),
                recommendation="This is target leakage: the column encodes the answer (recorded "
                "after the outcome, or derived from it). Remove it - any model trained with it "
                "will look excellent and fail in production.",
                columns=[c for c, _ in perfect],
                evidence={c: r for c, r in perfect},
            )
        )
    if suspicious:
        findings.append(
            Finding(
                check=CHECK,
                severity=Severity.HIGH,
                title=f"{len(suspicious)} feature(s) are suspiciously predictive alone",
                detail="; ".join(_describe(c, r) for c, r in suspicious),
                recommendation="Verify each is genuinely available at prediction time. "
                "If it is computed from, or after, the outcome, drop it.",
                columns=[c for c, _ in suspicious],
                evidence={c: r for c, r in suspicious},
            )
        )

    # ---- 3. names that reference the target -----------------------------
    flagged = {c for c, _ in perfect} | {c for c, _ in suspicious}
    ctx.excluded_features |= flagged
    t = str(ctx.target).lower()
    named = [
        c
        for c in ctx.feature_cols
        if t and t in str(c).lower() and c not in flagged and c not in id_set
    ]
    if named:
        findings.append(
            Finding(
                check=CHECK,
                severity=Severity.LOW,
                title=f"{len(named)} column name(s) reference the target '{ctx.target}'",
                detail=", ".join(named),
                recommendation="Check whether these were derived from the target.",
                columns=named,
            )
        )
    return findings
