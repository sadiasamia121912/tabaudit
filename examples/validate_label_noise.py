"""How well does the label-noise check recover *known* flipped labels on the demo data?

The demo generator knows which labels it flipped, so we can score the detector with
precision / recall.   Run:  python examples/validate_label_noise.py [noise_rate]

For the same measurement on ten real public datasets see benchmarks/evaluate.py and
benchmarks/sweep_label_noise.py - those are what set the thresholds.
"""

from __future__ import annotations

import sys

from tabaudit import run_audit
from tabaudit.demo import make_churn_dataset


def evaluate(noise_rate: float, seed: int = 7) -> dict:
    train, _, flipped = make_churn_dataset(seed=seed, noise_rate=noise_rate, return_truth=True)
    truth = set(train.index[flipped.to_numpy()])

    # leakage runs first so customer_id / churn_reason are excluded from the model's inputs
    report = run_audit(train, target="churn", checks=["leakage", "label_noise"])
    ev = next(f.evidence for f in report.findings if f.check == "label_noise")

    def pr(rows: list[int]) -> tuple[float, float]:
        hit = len(set(rows) & truth)
        return hit / max(1, len(rows)), hit / max(1, len(truth))

    return {
        "planted": len(truth),
        "planted_frac": len(truth) / len(train),
        "likely": len(ev["rows_likely"]),
        "likely_pr": pr(ev["rows_likely"]),
        "suspected": len(ev["rows"]),
        "suspected_pr": pr(ev["rows"]),
        "top25_precision": len(set(ev["rows"][:25]) & truth) / 25,
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
