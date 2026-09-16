"""The generated pipeline: right columns, right shape, and it actually runs."""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from tabaudit.cli import app
from tabaudit.demo import write_demo
from tabaudit.pipeline import column_groups, generate_pipeline

runner = CliRunner()
RNG = np.random.default_rng(3)


def mixed_frame(n: int = 200) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num": RNG.normal(size=n),
            "flag": RNG.random(n) > 0.5,
            "few": RNG.choice(["a", "b", "c"], n),
            "many": [f"v{i}" for i in range(n)],
            "when": pd.date_range("2024-01-01", periods=n, freq="D"),
            "y": RNG.integers(0, 2, n),
        }
    )


def test_column_groups_splits_by_dtype_then_cardinality():
    g = column_groups(mixed_frame(), target="y")
    assert g["numeric"] == ["num", "flag"]  # bool counts as numeric
    assert g["low_card"] == ["few"]
    assert g["high_card"] == ["many"]
    assert g["datetime"] == ["when"]
    assert not any("y" in cols for cols in g.values())


def test_excluded_columns_are_left_out_of_every_group():
    g = column_groups(mixed_frame(), target="y", exclude={"num": "leaky"})
    assert "num" not in g["numeric"]


def test_generated_code_is_valid_python_for_every_shape():
    df = mixed_frame()
    for target, task in (("y", "classification"), ("num", "regression"), (None, "unsupervised")):
        code = generate_pipeline(df, target, task, data_name="d.csv")
        compile(code, "<generated>", "exec")  # raises SyntaxError if not


def test_generated_code_is_tidy_enough_to_hand_to_someone():
    """It lands in the user's repo, so it should not be the thing that fails their linter."""
    code = generate_pipeline(mixed_frame(), "y", "classification", data_name="d.csv")
    assert max(len(line) for line in code.splitlines()) <= 88

    if importlib.util.find_spec("ruff") is None:  # pragma: no cover - dev extras missing
        pytest.skip("ruff not installed")
    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / "generated_pipeline.py"
        f.write_text(code, encoding="utf-8")
        for args in (["check", "--isolated"], ["format", "--isolated", "--check"]):
            done = subprocess.run(
                [sys.executable, "-m", "ruff", *args, str(f)], capture_output=True, text=True
            )
            assert done.returncode == 0, done.stdout + done.stderr


def test_the_pipeline_names_what_it_refuses_to_feed_the_model():
    df = mixed_frame()
    df["tabaudit_suspect"] = False
    code = generate_pipeline(
        df,
        "y",
        "classification",
        data_name="d.csv",
        exclude={"many": "leakage: predicts the target alone", "tabaudit_suspect": "marker"},
    )
    assert 'EXCLUDED = ["many", "tabaudit_suspect"]' in code
    assert "leakage: predicts the target alone" in code  # the reason travels with the column
    assert "HIGH_CARDINALITY = []" in code  # 'many' was the only one, and it is excluded


def test_imputers_appear_only_where_there_are_nulls():
    df = mixed_frame()
    clean = generate_pipeline(df, "y", "classification", data_name="d.csv")
    assert "SimpleImputer" not in clean

    df.loc[0, "num"] = np.nan
    with_nulls = generate_pipeline(df, "y", "classification", data_name="d.csv")
    assert 'SimpleImputer(strategy="median")' in with_nulls


def test_regression_target_gets_a_regressor():
    code = generate_pipeline(mixed_frame(), "num", "regression", data_name="d.csv")
    assert "HistGradientBoostingRegressor" in code
    assert "stratify" not in code  # meaningless for a continuous target


def test_fix_writes_a_pipeline_that_runs_on_the_cleaned_data(tmp_path):
    """The end of Phase 6: `fix` hands you code, and the code works."""
    train, test = write_demo(tmp_path)
    res = runner.invoke(
        app,
        [
            "fix",
            str(train),
            "-t",
            "churn",
            "--test",
            str(test),
            "-c",
            "schema,duplicates,leakage",
            "-q",
        ],
    )
    assert res.exit_code == 0, res.output

    generated = tmp_path / "churn_train_pipeline.py"
    assert generated.exists()
    code = generated.read_text(encoding="utf-8")
    assert "churn_reason" in code and "EXCLUDED" in code  # the leak is named, not silently kept

    run = subprocess.run(
        [sys.executable, str(generated)], cwd=tmp_path, capture_output=True, text=True, timeout=600
    )
    assert run.returncode == 0, run.stderr
    assert "held-out accuracy" in run.stdout and "5-fold" in run.stdout


def test_no_pipeline_skips_it(tmp_path):
    train, _ = write_demo(tmp_path)
    res = runner.invoke(app, ["fix", str(train), "-t", "churn", "-c", "schema", "--no-pipeline"])
    assert res.exit_code == 0, res.output
    assert not (tmp_path / "churn_train_pipeline.py").exists()
