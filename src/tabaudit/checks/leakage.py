"""Target leakage: identifier columns and single features that predict the target too well.

The core idea: a *single* feature that predicts the target almost perfectly is almost
never a real signal - it is a proxy for the answer that was recorded after the fact
(e.g. `churn_reason` filled in only for customers who churned).

Two rules are combined:

* **absolute** - a feature at AUC/R^2 >= 0.98 alone is a leak, full stop (CRITICAL);
* **relative** - below that, a leak is an *outlier*: it stands well above every other
  feature. Several strong features bunched together just mean the task is easy
  (e.g. cell-size measurements for a tumour dataset), not that any of them leaks.
  So the sorted single-feature scores are split at the biggest drop: features above a
  drop of >= GAP are "stand-alone" (HIGH, or MEDIUM if they are only moderately strong),
  features that are strong but part of the crowd are reported at INFO.
"""

from __future__ import annotations

import re
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
NEAR_PERFECT = 0.98  # alone, this is leakage no matter what the other features do
SUSPICIOUS = 0.90  # strong enough to be a leak *if* it also stands apart from the rest
SOFT = 0.75  # a "soft" leak: moderate on its own, but far above everything else
GAP = 0.15  # minimum drop to the next-best feature for a feature to count as stand-alone
MAX_STAND_ALONE = 2  # more strong features than this is a crowd (easy task), not a leak


def _looks_like_id_name(name: str) -> bool:
    low = str(name).lower()
    parts = low.replace("-", "_").split("_")
    return low.endswith("id") or any(p in ID_NAME_HINTS for p in parts)


def _words(name: str) -> list[str]:
    """Split a column name into lowercase words: 'workClass_id' -> ['work', 'class', 'id']."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))  # camelCase -> camel Case
    return [w for w in re.split(r"[^a-zA-Z0-9]+", spaced.lower()) if w]


def _mentions(col: str, target: str) -> bool:
    """True if the target name appears as a whole word in the column name.

    'class' matches 'class_of_service' but not 'workclass'.
    """
    return bool(target) and str(target).lower() in _words(col)


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
        # A leaf must never be required to hold more rows than half the rarest class, or the
        # tree cannot isolate that class at all (creditcard: 36 fraud rows in a 20k sample vs
        # the default leaf of 40 made a perfect leak score AUC 0.49).
        leaf = min(max(5, len(y_codes) // 500), max(2, min_class // 2))
        for col in feats:
            x = _fill_sentinel(Xe[col].to_numpy(dtype=float)).reshape(-1, 1)
            if np.unique(x).size < 2:
                continue
            model = DecisionTreeClassifier(
                max_depth=4, min_samples_leaf=leaf, random_state=ctx.random_state
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


def split_stand_alone(scores: dict[str, float], floor: float = 0.5) -> tuple[list[str], list[str]]:
    """Split features into (stand-alone, crowd) using the largest gap in sorted scores.

    Walk the scores from best to worst and find the biggest drop between neighbours
    (``floor`` = chance level stands in for the neighbour of the last feature). Only the
    first MAX_STAND_ALONE positions are considered: a leak is one or two columns, while a
    large group of strong features is an easy task. If the drop is >= GAP and starts at a
    feature scoring at least SOFT, everything above it "stands alone". Returns the
    stand-alone names (best first) and the remaining names that still clear SUSPICIOUS.
    """
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_drop, cut = 0.0, 0
    for i in range(min(MAX_STAND_ALONE, len(ranked))):
        top = ranked[i][1]
        if top < SOFT:
            break
        nxt = ranked[i + 1][1] if i + 1 < len(ranked) else floor
        drop = top - nxt
        if drop > best_drop:
            best_drop, cut = drop, i + 1
    if best_drop < GAP:
        cut = 0
    stand_alone = [c for c, _ in ranked[:cut]]
    crowd = [c for c, sc in ranked[cut:] if sc >= SUSPICIOUS]
    return stand_alone, crowd


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
    # Whether a value is *present* can leak on its own (Titanic `boat`: only survivors have
    # one). Score that directly and keep whichever view of the column is stronger; the
    # rules below then treat the column like any other.
    for col in ctx.feature_cols:
        if col in id_set or not ctx.df[col].isna().any():
            continue
        m_auc = _missingness_auc(ctx, col)
        if m_auc is None:
            continue
        if m_auc > scores.get(col, {"score": -np.inf})["score"]:
            scores[col] = {"metric": "AUC", "score": m_auc}
        scores[col]["missingness_auc"] = round(m_auc, 4)
    perfect = [c for c, r in scores.items() if r["score"] >= NEAR_PERFECT]
    rest = {c: r["score"] for c, r in scores.items() if c not in perfect}
    floor = 0.5 if ctx.task == "classification" else 0.0  # chance-level AUC / R^2
    stand_alone, crowd = split_stand_alone(rest, floor=floor)
    # A stand-alone feature is HIGH if it is strong in absolute terms, MEDIUM ("soft") if
    # it is only moderately strong but still far above everything else.
    suspicious = [c for c in stand_alone if rest[c] >= SUSPICIOUS]
    soft = [c for c in stand_alone if rest[c] < SUSPICIOUS]
    runner_up = max((sc for c, sc in rest.items() if c not in stand_alone), default=None)

    def _describe(col: str) -> str:
        r = scores[col]
        txt = f"{col} ({r['metric']}={r['score']:.3f})"
        if r.get("missingness_auc", 0) >= 0.9:
            txt += f" - its *missingness alone* has AUC {r['missingness_auc']:.2f}"
        return txt

    gap_note = f" Next-best feature scores {runner_up:.3f}." if runner_up is not None else ""

    if perfect:
        findings.append(
            Finding(
                check=CHECK,
                severity=Severity.CRITICAL,
                title=f"{len(perfect)} feature(s) predict the target almost perfectly on their own",
                detail="; ".join(_describe(c) for c in perfect),
                recommendation="This is target leakage: the column encodes the answer (recorded "
                "after the outcome, or derived from it). Remove it - any model trained with it "
                "will look excellent and fail in production.",
                columns=perfect,
                evidence={c: scores[c] for c in perfect},
            )
        )
    if suspicious:
        findings.append(
            Finding(
                check=CHECK,
                severity=Severity.HIGH,
                title=f"{len(suspicious)} feature(s) are suspiciously predictive alone",
                detail="; ".join(_describe(c) for c in suspicious)
                + f". Each stands far above every other feature.{gap_note}",
                recommendation="Verify each is genuinely available at prediction time. "
                "If it is computed from, or after, the outcome, drop it.",
                columns=suspicious,
                evidence={c: scores[c] for c in suspicious} | {"runner_up": runner_up},
            )
        )
    if soft:
        findings.append(
            Finding(
                check=CHECK,
                severity=Severity.MEDIUM,
                title=f"{len(soft)} feature(s) stand far above all others - possible soft leak",
                detail="; ".join(_describe(c) for c in soft)
                + f". Not near-perfect, but the gap to the next-best feature is >= {GAP:.2f}."
                + gap_note,
                recommendation="A single dominant feature is often something recorded during "
                "or after the outcome (e.g. call duration, days-in-hospital). Check when it "
                "becomes known; if that is after the prediction moment, drop it.",
                columns=soft,
                evidence={c: scores[c] for c in soft} | {"runner_up": runner_up},
            )
        )
    if crowd:
        findings.append(
            Finding(
                check=CHECK,
                severity=Severity.INFO,
                title=f"{len(crowd)} features each score >= {SUSPICIOUS:.2f} alone - "
                "highly separable task",
                detail="; ".join(_describe(c) for c in crowd)
                + ". They are bunched together (no single feature stands apart), which points "
                "to an easy task rather than a leak.",
                recommendation="No action needed unless the task should *not* be this easy; "
                "then check whether these columns are all derived from the same source as "
                "the target.",
                columns=crowd,
                evidence={c: scores[c] for c in crowd},
            )
        )

    # ---- 3. names that reference the target -----------------------------
    flagged = set(perfect) | set(suspicious) | set(soft)
    ctx.excluded_features |= flagged
    named = [
        c
        for c in ctx.feature_cols
        if _mentions(c, ctx.target) and c not in flagged and c not in id_set
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
