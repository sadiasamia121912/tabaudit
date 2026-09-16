"""Each check gets a tiny dataset where the answer is known."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabaudit import Severity, run_audit
from tabaudit.checks import duplicates, imbalance, impact, label_noise, leakage, schema
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
    # The copies are the appended rows 200..219 — the ones drop_duplicates() would remove.
    assert dup.evidence["rows"] == list(range(200, 220))


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


# ----------------------------------------------------------- label noise ----
def test_label_noise_exposes_flagged_rows():
    # Perfectly separable data, then flip 20 labels far from the boundary so the flips are
    # unambiguous. The full `rows` list is what the fault-injection benchmark scores against.
    # Own RNG: recall depends on the draw, and the shared RNG's state depends on test order.
    rng = np.random.default_rng(1)
    n = 600
    x1 = rng.normal(size=n)
    df = pd.DataFrame({"x1": x1, "x2": rng.normal(size=n), "y": (x1 > 0).astype(int)})
    flipped = df.index[df["x1"].abs() > 1.0][:20]
    df.loc[flipped, "y"] = 1 - df.loc[flipped, "y"]

    f = label_noise.run(ctx_for(df))
    assert len(f) == 1
    ev = f[0].evidence
    assert len(ev["rows"]) == ev["n_suspected"]
    assert len(ev["rows_likely"]) == ev["n_likely"]
    assert set(ev["rows_likely"]) <= set(ev["rows"])
    assert ev["rows"][: len(ev["top_suspects"])] == [s["row"] for s in ev["top_suspects"]]
    # With the self-confidence rule (4.4) the model's <20% belief in a planted label is enough
    # to flag it; the cleanlab filter this replaced recovered only 7-11 of these 20.
    recovered = set(ev["rows"]) & set(flipped)
    assert len(recovered) >= 17, f"only {len(recovered)}/20 planted flips were flagged"


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


def rare_class_frame(n: int = 20_000, n_pos: int = 36) -> pd.DataFrame:
    """A creditcard-like frame: 0.18% positives, three honest features."""
    rng = np.random.default_rng(5)
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, n_pos, replace=False)] = 1
    return pd.DataFrame(
        {
            "a": rng.normal(size=n) + 0.5 * y,
            "b": rng.normal(size=n),
            "c": rng.integers(0, 3, n),
            "y": y,
        }
    )


def test_leakage_missingness_leak_confined_to_rare_class():
    # Filled only for the 36 positives. The tree alone cannot isolate them (default leaf 40),
    # so this relies on the missingness-AUC path. Was missed 3/3 on creditcard before 4.4a.
    df = rare_class_frame()
    df["filled_for_pos"] = np.where(
        df["y"] == 1, np.random.default_rng(6).normal(size=len(df)), np.nan
    )
    f = leakage.run(ctx_for(df))
    hit = next(x for x in f if "filled_for_pos" in x.columns)
    assert hit.severity == Severity.CRITICAL
    assert "missingness alone" in hit.detail
    assert hit.evidence["filled_for_pos"]["missingness_auc"] == 1.0


def test_leakage_value_leak_confined_to_rare_class():
    # No NaNs at all: a flag that is 1 exactly for the positives. Only the tree can see it,
    # so this exercises the min_samples_leaf cap (36 positives < default leaf of 40).
    df = rare_class_frame()
    df["flag"] = df["y"]
    f = leakage.run(ctx_for(df))
    hit = next(x for x in f if "flag" in x.columns)
    assert hit.severity == Severity.CRITICAL
    assert not {"a", "b", "c"} & {c for x in f for c in x.columns}


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


def test_leakage_easy_task_is_info_not_high():
    # Six honest features that all predict well (like breast-w) - no single outlier.
    n = 800
    y = RNG.integers(0, 2, size=n)
    df = pd.DataFrame({f"m{i}": y * 2.2 + RNG.normal(size=n) for i in range(6)})
    df["y"] = y
    f = leakage.run(ctx_for(df))
    assert not any(x.severity in (Severity.CRITICAL, Severity.HIGH) for x in f)
    info = next(x for x in f if "highly separable" in x.title)
    assert info.severity == Severity.INFO and len(info.columns) == 6


def test_leakage_lone_strong_feature_is_high():
    # Same easy-ish signal, but only ONE feature carries it: it stands alone -> HIGH.
    n = 800
    y = RNG.integers(0, 2, size=n)
    df = pd.DataFrame({"m0": y * 2.2 + RNG.normal(size=n)})
    for i in range(1, 6):
        df[f"m{i}"] = y * 0.4 + RNG.normal(size=n)  # weak, honest
    df["y"] = y
    f = leakage.run(ctx_for(df))
    high = next(x for x in f if x.severity == Severity.HIGH)
    assert high.columns == ["m0"]
    assert high.evidence["runner_up"] < 0.75


def test_leakage_soft_leak_is_medium():
    # One feature far above the rest but below the 0.90 "suspicious" line (bank `duration`).
    n = 800
    y = RNG.integers(0, 2, size=n)
    df = pd.DataFrame({"duration": y * 1.4 + RNG.normal(size=n)})
    for i in range(5):
        df[f"m{i}"] = y * 0.2 + RNG.normal(size=n)
    df["y"] = y
    f = leakage.run(ctx_for(df))
    soft = next(x for x in f if "soft leak" in x.title)
    assert soft.severity == Severity.MEDIUM
    assert soft.columns == ["duration"]
    assert 0.75 <= soft.evidence["duration"]["score"] < 0.90


@pytest.mark.parametrize(
    "col, hit", [("workclass", False), ("class_of_service", True), ("SubClass", True)]
)
def test_leakage_target_name_match_is_whole_word(col, hit):
    df = base_frame(200).rename(columns={"y": "class"})
    df[col] = RNG.normal(size=len(df))
    f = leakage.run(ctx_for(df, target="class"))
    named = [x for x in f if "reference the target" in x.title]
    assert bool(named) is hit


def test_split_stand_alone():
    assert leakage.split_stand_alone({"a": 0.97, "b": 0.74, "c": 0.6}) == (["a"], [])
    assert leakage.split_stand_alone({"a": 0.97, "b": 0.95, "c": 0.6}) == (["a", "b"], [])
    crowd = {"a": 0.97, "b": 0.96, "c": 0.93, "d": 0.91, "e": 0.85}
    assert leakage.split_stand_alone(crowd) == ([], ["a", "b", "c", "d"])
    assert leakage.split_stand_alone({"a": 0.70, "b": 0.5}) == ([], [])  # below SOFT
    assert leakage.split_stand_alone({"a": 0.95}) == (["a"], [])  # lone feature vs chance
    assert leakage.split_stand_alone({}) == ([], [])


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


# ---------------------------------------------------------------- impact ----
def leaked_frame(n: int = 600) -> pd.DataFrame:
    df = base_frame(n)
    df["outcome_code"] = np.where(df["y"] == 1, "won", "lost")  # derived from the target
    return df


def test_impact_prices_the_flagged_columns():
    ctx = ctx_for(leaked_frame())
    leakage.run(ctx)  # this is what fills ctx.excluded_features
    f = impact.run(ctx)
    assert len(f) == 1
    assert f[0].severity == Severity.INFO  # evidence, not a second accusation
    assert "outcome_code" in f[0].columns
    ev = f[0].evidence
    assert ev["metric"] == "AUC"
    assert ev["with_flagged"] > ev["without_flagged"] + 0.05
    assert ev["delta"] == pytest.approx(ev["with_flagged"] - ev["without_flagged"], abs=1e-3)


def even_frame(n: int = 400) -> pd.DataFrame:
    """Two features of equal strength, so no single one stands apart and nothing is flagged."""
    x1, x2 = RNG.normal(size=n), RNG.normal(size=n)
    y = (x1 + x2 + RNG.normal(scale=0.8, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def test_impact_is_quiet_with_nothing_to_price():
    even = ctx_for(even_frame())
    leakage.run(even)
    assert not even.excluded_features  # precondition: the leakage check flagged nothing
    assert impact.run(even) == []
    # and it prices only what an earlier check flagged - alone it has no opinion
    assert impact.run(ctx_for(leaked_frame(300))) == []


def test_impact_prices_an_honest_dominant_feature_too():
    """The gap is the size of the bet, not proof of a leak: base_frame's x1 is legitimate
    (y is built from it) and gets priced exactly like a leak would be."""
    ctx = ctx_for(base_frame(300))
    leakage.run(ctx)
    assert "x1" in ctx.excluded_features  # MEDIUM "soft leak" - correct, given the gap
    f = impact.run(ctx)
    assert f and f[0].evidence["delta"] > 0
    assert "not proof of leakage" in f[0].recommendation


def test_impact_regression_reports_r2():
    n = 500
    x1, x2 = RNG.normal(size=n), RNG.normal(size=n)
    y = 3 * x1 + x2 + RNG.normal(scale=0.5, size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "copy_of_y": y + RNG.normal(scale=0.01, size=n), "y": y})
    ctx = ctx_for(df)
    leakage.run(ctx)
    f = impact.run(ctx)
    assert f and f[0].evidence["metric"] == "R2"
    assert f[0].evidence["with_flagged"] > f[0].evidence["without_flagged"]


def test_impact_never_moves_the_score():
    """The published benchmark scores must not shift because this check was added."""
    df = leaked_frame(400)
    with_impact = run_audit(df, target="y", checks=["leakage", "impact"])
    without = run_audit(df, target="y", checks=["leakage"])
    assert with_impact.score == without.score
    assert any(f.check == "impact" for f in with_impact.findings)


# ------------------------------------------------------- score breakdown ----
def test_score_breakdown_is_per_check_and_serialised():
    df = base_frame(300)
    df = pd.concat([df, df.head(30)], ignore_index=True)  # 30 exact duplicate rows
    report = run_audit(df, target="y", checks=["schema", "duplicates", "imbalance"])
    bd = report.score_breakdown
    assert list(bd) == ["schema", "duplicates", "imbalance"]  # registry order
    assert bd["duplicates"] < 100
    assert bd["schema"] == 100 and bd["imbalance"] == 100
    # the overall score is 100 minus every penalty, not the mean of the parts
    assert report.score == 100 - sum(100 - v for v in bd.values())
    assert report.to_dict()["score_breakdown"] == bd
