"""`tabaudit fix`: what it applies, what it refuses to guess at, and the CLI around it."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from tabaudit import run_audit
from tabaudit.cli import app
from tabaudit.demo import write_demo
from tabaudit.findings import AuditReport, CheckRun, DatasetSummary, Finding, Fix, Severity
from tabaudit.fix import apply_fixes

runner = CliRunner()
RNG = np.random.default_rng(7)


def messy_frame(n: int = 300) -> pd.DataFrame:
    """Every *safely* fixable defect at once: duplicates, a constant column, numbers stored
    as text, a leftover index column and a row with no label."""
    x1 = RNG.normal(size=n)
    df = pd.DataFrame(
        {
            "Unnamed: 0": np.arange(n),
            "x1": x1,
            "x2": RNG.normal(size=n),
            "charges": [f"{v:.2f}" for v in RNG.uniform(10, 99, size=n)],  # numeric as text
            "region": "eu-west",  # constant
            "y": (x1 + RNG.normal(scale=0.8, size=n) > 0).astype(int),
        }
    )
    df = pd.concat([df, df.head(12)], ignore_index=True)  # 12 exact duplicates
    df.loc[0, "y"] = np.nan  # one unlabeled row
    return df


def leaky_frame(n: int = 400) -> pd.DataFrame:
    df = pd.DataFrame({"x1": RNG.normal(size=n), "x2": RNG.normal(size=n)})
    df["y"] = (df.x1 + 0.5 * df.x2 + RNG.normal(scale=0.8, size=n) > 0).astype(int)
    df["outcome_code"] = np.where(df["y"] == 1, "won", "lost")  # derived from the target
    return df


def hand_report(finding: Finding, target: str | None = "y") -> AuditReport:
    """A report carrying one hand-made finding, for testing the applier's guards directly."""
    summary = DatasetSummary(
        path="<test>", n_rows=0, n_cols=0, target=target, task="classification", n_classes=2
    )
    return AuditReport(
        summary=summary,
        findings=[finding],
        checks=[CheckRun(finding.check, "ok", 0.0, 1)],
        version="0",
    )


# ------------------------------------------------------------------- 6.1 Fix ----
def test_fix_serialises_inside_a_finding():
    f = Finding(
        check="leakage",
        severity=Severity.CRITICAL,
        title="t",
        detail="d",
        fix=Fix("drop_columns", {"columns": ["a"]}, safe=False, flag="--drop-leaky"),
    )
    d = f.to_dict()
    assert d["fix"] == {
        "action": "drop_columns",
        "params": {"columns": ["a"]},
        "safe": False,
        "flag": "--drop-leaky",
    }
    assert json.loads(json.dumps(d))["fix"]["flag"] == "--drop-leaky"
    assert Finding(check="c", severity=Severity.LOW, title="t", detail="d").to_dict()["fix"] is None


def test_fix_rejects_nonsense():
    with pytest.raises(ValueError, match="Unknown fix action"):
        Fix("delete_everything", {})
    with pytest.raises(ValueError, match="must name the flag"):
        Fix("drop_columns", {"columns": ["a"]}, safe=False)


# --------------------------------------------------------------- 6.2 / 6.3 ----
def test_safe_fixes_are_applied_without_being_asked():
    df = messy_frame()
    report = run_audit(df, target="y", checks=["schema", "duplicates"])
    clean, plan = apply_fixes(df, report)

    assert "region" not in clean.columns  # constant
    assert "Unnamed: 0" not in clean.columns  # leftover index
    assert pd.api.types.is_numeric_dtype(clean["charges"])  # coerced
    assert clean.duplicated().sum() == 0
    assert clean["y"].isna().sum() == 0  # unlabeled row gone
    assert plan.rows_after < plan.rows_before and plan.cols_after < plan.cols_before
    assert plan.skipped == []
    assert all(s.safe for s in plan.applied)


def test_unsafe_fixes_are_skipped_and_named():
    df = leaky_frame()
    report = run_audit(df, target="y", checks=["leakage"])
    clean, plan = apply_fixes(df, report)

    assert "outcome_code" in clean.columns  # refused to guess
    assert plan.applied == []
    assert plan.flags_offered == ["--drop-leaky"]
    assert all(s.note == "needs --drop-leaky" for s in plan.skipped)


def test_the_flag_is_what_applies_it():
    df = leaky_frame()
    report = run_audit(df, target="y", checks=["leakage"])
    clean, plan = apply_fixes(df, report, ["--drop-leaky"])
    assert "outcome_code" not in clean.columns
    assert plan.flags_offered == []
    assert any(s.action == "drop_columns" and s.applied for s in plan.steps)


def test_flag_noise_marks_rows_and_removes_nothing():
    df = leaky_frame(600)
    df["y"] = np.where(RNG.random(len(df)) < 0.06, 1 - df["y"], df["y"])  # plant noise
    df = df.drop(columns=["outcome_code"])
    report = run_audit(df, target="y", checks=["leakage", "label_noise"])
    clean, _ = apply_fixes(df, report, ["--flag-noise"])

    assert len(clean) == len(df)  # never drops, never relabels
    assert clean["tabaudit_suspect"].dtype == bool
    assert 0 < clean["tabaudit_suspect"].sum() < len(df)
    assert (
        clean.loc[clean["tabaudit_suspect"], "y"] == df.loc[clean["tabaudit_suspect"], "y"]
    ).all()


def test_applying_the_plan_twice_is_a_noop():
    df = messy_frame()
    report = run_audit(df, target="y", checks=["schema", "duplicates"])
    once, _ = apply_fixes(df, report)
    twice, plan2 = apply_fixes(once, report)
    pd.testing.assert_frame_equal(once, twice)
    assert all(s.applied for s in plan2.steps)


def test_the_target_column_is_never_dropped():
    df = leaky_frame()
    finding = Finding(
        check="leakage",
        severity=Severity.CRITICAL,
        title="pretend the target itself was flagged",
        detail="d",
        columns=["y"],
        fix=Fix("drop_columns", {"columns": ["y", "outcome_code"]}, safe=False, flag="--x"),
    )
    clean, plan = apply_fixes(df, hand_report(finding), ["--x"])
    assert "y" in clean.columns
    assert "outcome_code" not in clean.columns
    assert plan.steps[0].columns == ["outcome_code"]


def test_a_fix_that_would_empty_the_dataset_is_skipped():
    df = pd.DataFrame({"x1": RNG.normal(size=80)})
    df["y"] = (df.x1 > 0).astype(int)
    finding = Finding(
        check="leakage",
        severity=Severity.CRITICAL,
        title="the only feature is leaky",
        detail="d",
        columns=["x1"],
        fix=Fix("drop_columns", {"columns": ["x1"]}, safe=False, flag="--x"),
    )
    clean, plan = apply_fixes(df, hand_report(finding), ["--x"])
    assert list(clean.columns) == ["x1", "y"]
    assert plan.skipped[0].note == "would leave no feature columns"


# ------------------------------------------------------------------ 6.4 CLI ----
def test_fix_command_writes_data_and_plan(tmp_path):
    train, test = write_demo(tmp_path)
    res = runner.invoke(
        app, ["fix", str(train), "-t", "churn", "--test", str(test), "-c", "schema,duplicates"]
    )
    assert res.exit_code == 0, res.output

    clean = tmp_path / "churn_train.clean.csv"
    plan_file = tmp_path / "churn_train.fixplan.json"
    assert clean.exists() and plan_file.exists()
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    assert plan["rows_after"] < plan["rows_before"]
    assert plan["n_applied"] >= 1
    assert pd.read_csv(clean).duplicated().sum() == 0


def test_fix_exits_zero_with_unsafe_fixes_outstanding(tmp_path):
    """Skipping a fix the user did not authorise is the correct outcome, not a failure."""
    train, _ = write_demo(tmp_path)
    res = runner.invoke(app, ["fix", str(train), "-t", "churn", "-c", "leakage"])
    assert res.exit_code == 0, res.output
    assert "--drop-leaky" in res.output
    plan = json.loads((tmp_path / "churn_train.fixplan.json").read_text(encoding="utf-8"))
    assert plan["n_applied"] == 0 and plan["n_skipped"] >= 1
    assert plan["flags_offered"] == ["--drop-leaky"]
    # the leaky column is still in the written file
    assert "churn_reason" in pd.read_csv(tmp_path / "churn_train.clean.csv").columns


def test_fix_honours_out_and_keeps_the_input_untouched(tmp_path):
    train, _ = write_demo(tmp_path)
    before = train.read_bytes()
    out = tmp_path / "sub" / "clean.parquet"
    res = runner.invoke(
        app, ["fix", str(train), "-t", "churn", "-c", "schema", "--out", str(out), "-q"]
    )
    assert res.exit_code == 0, res.output
    assert out.exists() and train.read_bytes() == before
    assert len(pd.read_parquet(out)) > 0
