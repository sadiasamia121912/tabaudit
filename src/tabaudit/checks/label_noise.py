"""Likely mislabeled rows, found from an out-of-fold model's self-confidence.

We train a gradient-boosting model with cross-validation so every row is predicted by a
model that never saw it, and look at the probability that model gives the row's *given*
label ("self-confidence"). A low value means the model, having learned the pattern from
the other rows, confidently disagrees with the label.

Two tiers, both plain thresholds on self-confidence: likely (< 0.2) and suspected (< 0.3).
The thresholds - and the decision *not* to filter further with cleanlab's confident-learning
step - were set by benchmarks/sweep_label_noise.py on 3% planted flips across 10 public
datasets: cleanlab's filter left precision unchanged and cost 7 points of recall.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from tabaudit.context import AuditContext
from tabaudit.findings import Finding, Severity

CHECK = "label_noise"
TOP_N = 25
# A row is *likely* mislabeled when the out-of-fold model gives its given label less than
# this probability, *suspected* below the looser value. Set by benchmarks/sweep_label_noise.py
# (see docs/checks.md): 0.2 had the best precision/recall trade-off (0.80 / 0.64 on planted
# flips) without changing any real dataset's severity; 0.3 adds recall (0.73) at 0.70
# precision and would have pushed four of ten public datasets to HIGH, so it is the
# broader tier, not the headline.
LIKELY_MAX_SELF_CONFIDENCE = 0.2
SUSPECTED_MAX_SELF_CONFIDENCE = 0.3


def make_model(random_state: int) -> HistGradientBoostingClassifier:
    """Deliberately regularised: an over-confident model calls its own mistakes label errors."""
    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=random_state,
    )


def out_of_fold_probs(
    ctx: AuditContext,
) -> tuple[pd.Index, np.ndarray, np.ndarray, np.ndarray] | None:
    """The expensive half of the check: (row index, y codes, class names, out-of-fold
    predicted probabilities), or None when the data cannot support it. Shared with
    benchmarks/sweep_label_noise.py so every flagging rule is compared on the same model.
    Leaky / identifier columns must already be in ctx.excluded_features (the leakage check
    does this): a leak would make the model agree with every wrong label and hide the noise."""
    if ctx.task != "classification" or ctx.y is None:
        return None
    idx = ctx.sample_index()
    Xe = ctx.X_encoded_clean.loc[idx]
    y = ctx.y.loc[idx]
    keep = y.notna().to_numpy()
    Xe, y = Xe[keep], y[keep]
    if len(y) < 50 or Xe.shape[1] == 0:
        return None

    y_codes, classes = pd.factorize(y)
    counts = np.bincount(y_codes)
    if len(classes) < 2 or counts.min() < 5:
        return None
    n_splits = int(min(5, counts.min()))

    model = make_model(ctx.random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=ctx.random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_probs = cross_val_predict(model, Xe.to_numpy(), y_codes, cv=cv, method="predict_proba")
    return Xe.index, y_codes, np.asarray(classes), pred_probs


def run(ctx: AuditContext) -> list[Finding]:
    oof = out_of_fold_probs(ctx)
    if oof is None:
        return []
    index, y_codes, classes, pred_probs = oof
    self_conf = pred_probs[np.arange(len(y_codes)), y_codes]
    suspected = self_conf < SUSPECTED_MAX_SELF_CONFIDENCE
    likely = self_conf < LIKELY_MAX_SELF_CONFIDENCE
    # Rank all suspects by how little the model believes the given label.
    issue_idx = np.flatnonzero(suspected)
    issue_idx = issue_idx[np.argsort(self_conf[issue_idx])]

    n_suspected, n_likely = int(suspected.sum()), int(likely.sum())
    if n_suspected == 0:
        return []
    n = len(y_codes)
    frac_likely = n_likely / n
    frac_suspected = n_suspected / n
    if frac_likely >= 0.08:
        sev = Severity.HIGH
    elif frac_likely >= 0.03:
        sev = Severity.MEDIUM
    elif frac_likely >= 0.005:
        sev = Severity.LOW
    else:
        sev = Severity.INFO

    # Every flagged row, most-confident first. The report shows the top TOP_N; the full lists
    # exist so the benchmark can score precision/recall against planted label flips.
    rows_suspected = [int(index[i]) for i in issue_idx]
    rows_likely = [int(index[i]) for i in issue_idx if likely[i]]

    # Top suspects, for the report.
    suspects = []
    for i in issue_idx[:TOP_N]:
        given = y_codes[i]
        suggested = int(np.argmax(pred_probs[i]))
        suspects.append(
            {
                "row": int(index[i]),
                "given_label": str(classes[given]),
                "suggested_label": str(classes[suggested]),
                "confidence_in_given": round(float(pred_probs[i, given]), 3),
                "confidence_in_suggested": round(float(pred_probs[i, suggested]), 3),
            }
        )
    scope = f" (of a {n:,}-row sample)" if len(ctx.df) > ctx.max_rows else ""
    excluded = sorted(ctx.excluded_features & set(ctx.X_encoded.columns))
    if excluded:
        scope += f"; estimated with {', '.join(excluded)} excluded as leaky/identifier"
    return [
        Finding(
            check=CHECK,
            severity=sev,
            title=f"~{n_likely:,} rows ({frac_likely:.1%}) are likely mislabeled, "
            f"{n_suspected - n_likely:,} more suspected",
            detail="An out-of-fold model confidently disagrees with the given label on these rows"
            f"{scope}. 'Likely' = model gives the given label <{LIKELY_MAX_SELF_CONFIDENCE:.0%} "
            f"probability; 'suspected' = <{SUSPECTED_MAX_SELF_CONFIDENCE:.0%}. Label noise "
            "caps achievable accuracy and misleads model selection.",
            recommendation="Review the top suspects below (ranked most-confident first). If they are "
            "real errors, relabel or drop them; do not silently trust the reported test accuracy.",
            columns=[ctx.target or ""],
            evidence={
                "n_likely": n_likely,
                "n_suspected": n_suspected,
                "fraction_likely": round(frac_likely, 4),
                "fraction_suspected": round(frac_suspected, 4),
                "top_suspects": suspects,
                "rows": rows_suspected,
                "rows_likely": rows_likely,
            },
        )
    ]
