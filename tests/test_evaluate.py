"""The evaluation harness on a synthetic frame: no network, a few seconds, and it must
produce a fully populated result for every fault type."""

from __future__ import annotations

import numpy as np
import pandas as pd

import evaluate


def frame(n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = (x1 + 0.5 * x2 + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": rng.choice(list("abc"), n), "y": y})


def test_evaluate_dataset_scores_every_fault():
    results = evaluate.evaluate_dataset(
        "toy", frame(), "y", seeds=[0], faults=list(evaluate.FAULTS)
    )
    by_fault = {r["fault"]: r for r in results}
    assert set(by_fault) == set(evaluate.FAULTS)
    assert all("error" not in r for r in results)

    assert by_fault["duplicates"]["recall"] == 1.0
    assert by_fault["duplicates"]["precision"] == 1.0

    flips = by_fault["label_flips"]
    assert flips["n_in_sample"] == flips["n_planted_total"] == 24  # 3% of 800, nothing sampled away
    for tier in ("suspected", "likely"):
        assert {"precision", "recall", "precision_excl_baseline"} <= set(flips[tier])

    for fault in ("leak_copy", "leak_missingness"):
        r = by_fault[fault]
        assert r["detected"] is True
        assert r["severity"] == "critical"
        # FPR is not asserted to be 0: on a 3-feature toy the gap rule flags the real features
        # as a "soft leak" (they do stand far above `cat`). Real datasets measure FPR; here we
        # only check the bookkeeping is right.
        assert 0.0 <= r["fpr"] <= 1.0
        assert set(r["false_positive_columns"]) <= {"x1", "x2", "cat"}
        assert r["n_innocent_columns"] == 3


def test_leak_flagged_ignores_info_and_identifier_findings():
    from tabaudit.findings import AuditReport, DatasetSummary, Finding, Severity

    findings = [
        Finding("leakage", Severity.CRITICAL, "1 feature(s) predict ...", "", columns=["leak"]),
        Finding("leakage", Severity.INFO, "3 features each score ...", "", columns=["a", "b"]),
        Finding("leakage", Severity.MEDIUM, "1 identifier-like column(s) ...", "", columns=["id"]),
        Finding("schema", Severity.HIGH, "constant column", "", columns=["c"]),
    ]
    summary = DatasetSummary("x", 1, 1, "y", "classification", 2)
    report = AuditReport(summary=summary, findings=findings, checks=[], version="0")
    assert evaluate.leak_flagged(report) == {"leak"}
