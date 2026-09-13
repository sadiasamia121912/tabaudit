"""Each check gets a tiny dataset where the answer is known."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabaudit import Severity, run_audit
from tabaudit.checks import duplicates, imbalance, leakage, schema
from tabaudit.context import AuditContext
from tabaudit.loader import infer_task

RNG = np.random.default_rng(0)


def ctx_for(
    df: pd.DataFrame, target: str | None = "y", test: pd.DataFrame | None = None
) -> AuditContext:
    task = infer_task(df[target]) if target else "unsupervised"
    return AuditContext(df=df, target=target, task=task, test_df=test)


def base_frame(n: int = 400) -> pd.DataFrame:
    x1 = RNG.normal(size=n)
    x2 = RNG.normal(size=n)
    y = (x1 + 0.5 * x2 + RNG.normal(scale=0.8, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


# ---------------------------------------------------------------- schema ----
def test_schema_flags_missing_constant_and_numeric_text():
    df = base_frame()
    df["mostly_missing"] = np.where(RNG.random(len(df)) < 0.7, np.nan, 1.0)
    df["const"] = "same"
    df["num_as_text"] = RNG.integers(0, 100, len(df)).astype(str)
    f = schema.run(ctx_for(df))
    titles = " | ".join(x.title for x in f)
    assert ">=50% missing" in titles
    assert "constant column" in titles
    assert "stored as text" in titles


def test_schema_clean_frame_is_quiet():
    assert schema.run(ctx_for(base_frame())) == []


# ------------------------------------------------------------ duplicates ----
def test_duplicates_exact_rows():
    df = base_frame(200)
    df = pd.concat([df, df.head(20)], ignore_index=True)  # 10% dups -> HIGH
    f = duplicates.run(ctx_for(df))
    dup = next(x for x in f if "exact duplicate" in x.title)
    assert dup.severity == Severity.HIGH
    assert dup.evidence["n_duplicates"] == 20


def test_duplicates_conflicting_labels():
    df = base_frame(200)
    twin = df.head(5).copy()
    twin["y"] = 1 - twin["y"]
    df = pd.concat([df, twin], ignore_index=True)
    f = duplicates.run(ctx_for(df))
    assert any("conflicting labels" in x.title for x in f)


def test_duplicates_train_test_overlap_is_critical():
    train = base_frame(300)
    test = pd.concat([base_frame(100), train.sample(10, random_state=1)], ignore_index=True)
    f = duplicates.run(ctx_for(train, test=test))
    ov = next(x for x in f if "also appear in the training set" in x.title)
    assert ov.severity == Severity.CRITICAL
    assert ov.evidence["n_overlap"] == 10


# ------------------------------------------------------------- imbalance ----
@pytest.mark.parametrize(
    "ratio,expected", [(2, None), (5, Severity.LOW), (20, Severity.MEDIUM), (200, Severity.HIGH)]
)
def test_imbalance_severity_scales_with_ratio(ratio, expected):
    n_min = 50
    y = np.array([0] * (n_min * ratio) + [1] * n_min)
    df = pd.DataFrame({"x": RNG.normal(size=len(y)), "y": y})
    f = imbalance.run(ctx_for(df))
    sev = next((x.severity for x in f if "Class imbalance" in x.title), None)
    assert sev == expected


def test_imbalance_skips_regression():
    df = pd.DataFrame({"x": RNG.normal(size=100), "y": RNG.normal(size=100)})
    assert imbalance.run(ctx_for(df)) == []


# --------------------------------------------------------------- leakage ----
def test_leakage_catches_perfect_proxy_and_id():
    df = base_frame(600)
    df["outcome_code"] = np.where(df["y"] == 1, "won", "lost")  # derived from target
    df["row_id"] = np.arange(len(df))
    f = leakage.run(ctx_for(df))
    perfect = next(x for x in f if "almost perfectly" in x.title)
    assert perfect.severity == Severity.CRITICAL
    assert "outcome_code" in perfect.columns
    ids = next(x for x in f if "identifier-like" in x.title)
    assert "row_id" in ids.columns


def test_leakage_catches_missingness_leak():
    df = base_frame(600)
    df["note"] = np.where(df["y"] == 1, "x", None)  # present iff positive
    f = leakage.run(ctx_for(df))
    perfect = next(x for x in f if "almost perfectly" in x.title)
    assert "note" in perfect.columns
    assert "missingness" in perfect.detail


def test_leakage_does_not_flag_honest_features():
    f = leakage.run(ctx_for(base_frame(600)))
    assert not any(x.severity in (Severity.CRITICAL, Severity.HIGH) for x in f)


def test_leakage_regression_r2():
    n = 600
    x = RNG.normal(size=n)
    df = pd.DataFrame({"x": x, "y": 3 * x + RNG.normal(scale=0.5, size=n)})
    df["y_copy"] = df["y"] * 2 + 1
    f = leakage.run(ctx_for(df))
    perfect = next(x for x in f if "almost perfectly" in x.title)
    assert "y_copy" in perfect.columns
    assert perfect.evidence["y_copy"]["metric"] == "R2"


# ------------------------------------------------------ end-to-end / API ----
def test_run_audit_on_dataframe_scores_and_serialises():
    report = run_audit(base_frame(300), target="y", checks=["schema", "duplicates", "imbalance"])
    assert report.score == 100 and report.grade == "A"
    d = report.to_dict()
    assert d["summary"]["task"] == "classification"
    assert {c["name"] for c in d["checks"]} == {"schema", "duplicates", "imbalance"}


def test_run_audit_unknown_target_gives_hint():
    with pytest.raises(KeyError, match="Did you mean 'y'"):
        run_audit(base_frame(50), target="Y")


def test_run_audit_unknown_check():
    with pytest.raises(ValueError, match="Unknown check"):
        run_audit(base_frame(50), target="y", checks=["nope"])


def test_demo_dataset_fails_hard():
    from tabaudit.demo import make_churn_dataset

    train, test = make_churn_dataset(n=1500)
    report = run_audit(train, target="churn", test=test)
    assert report.grade in ("D", "F")
    crit = [f for f in report.findings if f.severity == Severity.CRITICAL]
    assert any("churn_reason" in f.columns for f in crit)  # the planted leak
    assert any(f.check == "label_noise" for f in report.findings)
