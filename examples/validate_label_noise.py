"""How well does the label-noise check recover *known* flipped labels?

The demo generator knows which labels it flipped, so we can score the detector with
precision / recall.   Run:  python examples/validate_label_noise.py [noise_rate]
"""

from __future__ import annotations

import sys

import numpy as np
from cleanlab.filter import find_label_issues
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from tabaudit.checks.label_noise import LIKELY_MAX_SELF_CONFIDENCE, make_model
from tabaudit.context import AuditContext
from tabaudit.demo import make_churn_dataset
from tabaudit.loader import infer_task


def evaluate(noise_rate: float, seed: int = 7) -> dict:
    train, _, flipped = make_churn_dataset(seed=seed, noise_rate=noise_rate, return_truth=True)
    flipped = flipped.to_numpy()

    ctx = AuditContext(df=train, target="churn", task=infer_task(train["churn"]))
    ctx.excluded_features |= {"customer_id", "churn_reason"}  # what the leakage check would do
    X = ctx.X_encoded_clean.to_numpy()
    y = train["churn"].to_numpy()

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    probs = cross_val_predict(make_model(42), X, y, cv=cv, method="predict_proba")
    suspected = find_label_issues(
        labels=y, pred_probs=probs, filter_by="confident_learning", n_jobs=1
    )
    self_conf = probs[np.arange(len(y)), y]
    likely = suspected & (self_conf < LIKELY_MAX_SELF_CONFIDENCE)
    ranked = np.flatnonzero(suspected)
    ranked = ranked[np.argsort(self_conf[ranked])]

    def pr(mask):
        tp = int((mask & flipped).sum())
        return tp / max(1, mask.sum()), tp / max(1, flipped.sum())

    return {
        "planted": int(flipped.sum()),
        "planted_frac": float(flipped.mean()),
        "likely": int(likely.sum()),
        "likely_pr": pr(likely),
        "suspected": int(suspected.sum()),
        "suspected_pr": pr(suspected),
        "top25_precision": float(flipped[ranked[:25]].mean()),
    }


if __name__ == "__main__":
    rate = float(sys.argv[1]) if len(sys.argv) > 1 else 0.06
    r = evaluate(rate)
    print(f"planted flips     : {r['planted']:>5}  ({r['planted_frac']:.1%})")
    print(
        f"likely mislabeled : {r['likely']:>5}  precision {r['likely_pr'][0]:.0%}  recall {r['likely_pr'][1]:.0%}"
    )
    print(
        f"suspected         : {r['suspected']:>5}  precision {r['suspected_pr'][0]:.0%}  recall {r['suspected_pr'][1]:.0%}"
    )
    print(f"top-25 precision  : {r['top25_precision']:.0%}")
