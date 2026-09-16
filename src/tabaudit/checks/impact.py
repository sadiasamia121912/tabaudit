"""What the flagged columns are worth: held-out performance with and without them.

The `leakage` check says a column looks dangerous. This check says how much of the
dataset's apparent performance rests on it: one cross-validated model on every feature,
one on the same rows and folds minus whatever `leakage` excluded (leaky and identifier-like
columns), and the difference between the two scores.

Read the gap as *the size of the bet*, not as proof: a genuinely legitimate strong feature
moves the number exactly the same way. What it tells you is what the reported score would
become if the flagged columns turned out to be unavailable at prediction time - which is
the question "is this a leak?" actually costs you an answer to.

The finding is always INFO: the defect itself is already reported, and scored, by the
check that flagged the columns. This one only prices it, so it must not move the score.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

from tabaudit.checks.label_noise import make_model
from tabaudit.context import AuditContext
from tabaudit.findings import Finding, Severity

CHECK = "impact"
MIN_ROWS = 50
# Below this the two models are the same model as far as anyone should care.
NEGLIGIBLE = 0.005
MAX_NAMED = 10  # columns listed by name in the finding text


def _cv_score(
    Xe: pd.DataFrame, y: pd.Series, task: str, random_state: int
) -> tuple[float, int] | None:
    """Out-of-fold AUC (classification) or R2 (regression), and the fold count."""
    if Xe.shape[1] == 0 or len(y) < MIN_ROWS:
        return None
    X = Xe.to_numpy()

    if task == "classification":
        y_codes, classes = pd.factorize(y)
        counts = np.bincount(y_codes)
        if len(classes) < 2 or counts.min() < 5:
            return None
        n_splits = int(min(5, counts.min()))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probs = cross_val_predict(
                make_model(random_state), X, y_codes, cv=cv, method="predict_proba"
            )
        if len(classes) == 2:
            score = roc_auc_score(y_codes, probs[:, 1])
        else:
            score = roc_auc_score(y_codes, probs, multi_class="ovr", average="macro")
        return float(score), n_splits

    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.isna().any():
        return None
    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=random_state,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = cross_val_predict(model, X, y_num.to_numpy(), cv=cv)
    return float(r2_score(y_num, pred)), 5


def run(ctx: AuditContext) -> list[Finding]:
    if ctx.target is None or ctx.task == "unsupervised" or ctx.y is None:
        return []
    # Whatever the leakage check took out of play. Nothing flagged -> nothing to price.
    flagged = sorted(ctx.excluded_features & set(ctx.X_encoded.columns))
    if not flagged:
        return []

    idx = ctx.sample_index()
    y = ctx.y.loc[idx]
    keep = y.notna().to_numpy()
    y = y[keep]
    full = ctx.X_encoded.loc[idx][keep]
    clean = ctx.X_encoded_clean.loc[idx][keep]
    if clean.shape[1] == 0:
        return []  # every feature was flagged; "without" is not a dataset

    with_flagged = _cv_score(full, y, ctx.task, ctx.random_state)
    without_flagged = _cv_score(clean, y, ctx.task, ctx.random_state)
    if with_flagged is None or without_flagged is None:
        return []
    (kept_score, n_splits), (dropped_score, _) = with_flagged, without_flagged
    delta = kept_score - dropped_score
    metric = "AUC" if ctx.task == "classification" else "R2"

    named = ", ".join(flagged[:MAX_NAMED])
    if len(flagged) > MAX_NAMED:
        named += f", +{len(flagged) - MAX_NAMED} more"

    if delta >= NEGLIGIBLE:
        detail = (
            f"{delta:.3f} of the apparent {metric} rests on {named}. Without them the same "
            f"model scores {dropped_score:.3f}."
        )
        recommendation = (
            f"If those columns are not genuinely available at prediction time, {dropped_score:.3f} "
            f"is the honest expectation for this dataset, not {kept_score:.3f}. A large gap is "
            "not proof of leakage - a legitimately strong feature looks identical here - so "
            "decide per column when each value becomes known."
        )
    elif delta <= -NEGLIGIBLE:
        detail = (
            f"Dropping {named} *raised* the held-out {metric} from {kept_score:.3f} to "
            f"{dropped_score:.3f}: as model inputs they cost more in noise than they add."
        )
        recommendation = (
            "Drop them. They are not paying for themselves even before the question of "
            "leakage comes up."
        )
    else:
        detail = (
            f"Dropping {named} changes the held-out {metric} by {abs(delta):.3f} "
            f"({kept_score:.3f} -> {dropped_score:.3f}) - nothing. The remaining features "
            "carry the same signal."
        )
        recommendation = (
            "Drop them. Whatever they are, the dataset does not need them, so the safe "
            "choice is free."
        )

    return [
        Finding(
            check=CHECK,
            severity=Severity.INFO,
            title=(
                f"Held-out {metric} {kept_score:.3f} with the {len(flagged)} flagged "
                f"column(s), {dropped_score:.3f} without"
            ),
            detail=detail,
            recommendation=recommendation,
            columns=flagged,
            evidence={
                "metric": metric,
                "with_flagged": round(kept_score, 4),
                "without_flagged": round(dropped_score, 4),
                "delta": round(delta, 4),
                "n_rows": len(y),
                "n_features_dropped": len(flagged),
                "n_splits": n_splits,
            },
        )
    ]
