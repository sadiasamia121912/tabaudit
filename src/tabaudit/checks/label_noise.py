"""Likely mislabeled rows, found with confident learning (cleanlab).

We train an out-of-fold gradient-boosting model, obtain predicted class probabilities for
every row, and let cleanlab compare them with the given labels. Rows whose given label is
confidently contradicted by the model are flagged.
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
# A suspect counts as *likely* mislabeled when the out-of-fold model gives the given label
# less than this probability. Chosen on the synthetic benchmark (examples/validate_label_noise.py)
# as the point where precision is useful without collapsing recall.
LIKELY_MAX_SELF_CONFIDENCE = 0.2


def make_model(random_state: int) -> HistGradientBoostingClassifier:
    """Deliberately regularised: an over-confident model makes cleanlab over-flag."""
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


def run(ctx: AuditContext) -> list[Finding]:
    if ctx.task != "classification" or ctx.y is None:
        return []
    try:
        from cleanlab.filter import find_label_issues
    except ImportError:  # pragma: no cover
        return []

    idx = ctx.sample_index()
    # Leaky / identifier columns are excluded on purpose: a leak would make the model agree
    # with every wrong label and hide the noise we are trying to find.
    Xe = ctx.X_encoded_clean.loc[idx]
    y = ctx.y.loc[idx]
    keep = y.notna().to_numpy()
    Xe, y = Xe[keep], y[keep]
    if len(y) < 50 or Xe.shape[1] == 0:
        return []

    y_codes, classes = pd.factorize(y)
    counts = np.bincount(y_codes)
    if len(classes) < 2 or counts.min() < 5:
        return []
    n_splits = int(min(5, counts.min()))

    model = make_model(ctx.random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=ctx.random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_probs = cross_val_predict(model, Xe.to_numpy(), y_codes, cv=cv, method="predict_proba")
        # n_jobs=1: cleanlab's worker pool misbehaves on Windows and buys nothing at this size.
        suspected = find_label_issues(
            labels=y_codes, pred_probs=pred_probs, filter_by="confident_learning", n_jobs=1
        )
    self_conf = pred_probs[np.arange(len(y_codes)), y_codes]
    likely = suspected & (self_conf < LIKELY_MAX_SELF_CONFIDENCE)
    # Rank all suspects by how little the model believes the given label.
    issue_idx = np.flatnonzero(suspected)
    issue_idx = issue_idx[np.argsort(self_conf[issue_idx])]

    n_suspected, n_likely = int(suspected.sum()), int(likely.sum())
    if n_suspected == 0:
        return []
    frac_likely = n_likely / len(y)
    frac_suspected = n_suspected / len(y)
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
    rows_suspected = [int(Xe.index[i]) for i in issue_idx]
    rows_likely = [int(Xe.index[i]) for i in issue_idx if likely[i]]

    # Top suspects, for the report.
    suspects = []
    for i in issue_idx[:TOP_N]:
        given = y_codes[i]
        suggested = int(np.argmax(pred_probs[i]))
        suspects.append(
            {
                "row": int(Xe.index[i]),
                "given_label": str(classes[given]),
                "suggested_label": str(classes[suggested]),
                "confidence_in_given": round(float(pred_probs[i, given]), 3),
                "confidence_in_suggested": round(float(pred_probs[i, suggested]), 3),
            }
        )
    scope = f" (of a {len(y):,}-row sample)" if len(idx) < len(ctx.df) else ""
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
            "probability; 'suspected' = flagged by confident learning (cleanlab). Label noise "
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
