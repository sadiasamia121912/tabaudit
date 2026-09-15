"""Injectors must plant exactly what they promise, leave the input untouched, and be
reproducible from the seed. The last two tests check the contract that matters most:
what an injector reports as truth is what the corresponding check flags."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inject import (
    INJECTORS,
    LEAK_COPY_COL,
    LEAK_MISSING_COL,
    inject_duplicates,
    inject_label_flips,
    inject_leak_copy,
    inject_leak_missingness,
)
from tabaudit.checks import duplicates, leakage
from tabaudit.context import AuditContext


def frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = (x1 + 0.5 * x2 + rng.normal(scale=0.8, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": rng.choice(list("abc"), n), "y": y})


def ctx_for(df: pd.DataFrame) -> AuditContext:
    return AuditContext(df=df, target="y", task="classification")


# ------------------------------------------------------------------ purity ----
@pytest.mark.parametrize("name", list(INJECTORS))
def test_injectors_do_not_mutate_input_and_are_seeded(name):
    df = frame()
    before = df.copy()
    a = INJECTORS[name](df, "y", np.random.default_rng(7))
    b = INJECTORS[name](df, "y", np.random.default_rng(7))
    pd.testing.assert_frame_equal(df, before)
    pd.testing.assert_frame_equal(a.df, b.df)
    assert (a.rows, a.columns) == (b.rows, b.columns)
    assert a.kind == name


# -------------------------------------------------------------- duplicates ----
def test_inject_duplicates_appends_exact_copies():
    df = frame(400)
    inj = inject_duplicates(df, "y", np.random.default_rng(0), frac=0.05)
    assert len(inj.df) == 420
    assert inj.rows == list(range(400, 420))
    # Every planted row is an exact copy of some original row.
    merged = inj.df.loc[inj.rows].merge(df, how="left", indicator=True)
    assert (merged["_merge"] == "both").all()
    pd.testing.assert_frame_equal(inj.df.iloc[:400], df)


# ------------------------------------------------------------- label flips ----
def test_inject_label_flips_changes_exactly_the_reported_rows():
    df = frame(400)
    inj = inject_label_flips(df, "y", np.random.default_rng(0), frac=0.03)
    changed = df.index[df["y"] != inj.df["y"]].tolist()
    assert changed == inj.rows
    assert len(changed) == 12
    assert set(inj.df["y"].unique()) <= set(df["y"].unique())  # still valid classes
    pd.testing.assert_frame_equal(inj.df.drop(columns="y"), df.drop(columns="y"))


def test_inject_label_flips_multiclass_never_keeps_same_label():
    df = frame(300)
    df["y"] = np.random.default_rng(1).integers(0, 4, len(df))
    inj = inject_label_flips(df, "y", np.random.default_rng(0), frac=0.1)
    assert (df.loc[inj.rows, "y"] != inj.df.loc[inj.rows, "y"]).all()


# ----------------------------------------------------------------- leaks ----
def test_inject_leak_copy_tracks_target():
    df = frame()
    inj = inject_leak_copy(df, "y", np.random.default_rng(0), noise=0.1)
    assert inj.columns == [LEAK_COPY_COL]
    assert inj.df[LEAK_COPY_COL].corr(inj.df["y"]) > 0.95
    pd.testing.assert_frame_equal(inj.df.drop(columns=LEAK_COPY_COL), df)


def test_inject_leak_missingness_is_filled_for_one_class_only():
    df = frame()
    inj = inject_leak_missingness(df, "y", np.random.default_rng(0))
    rarest = df["y"].value_counts().idxmin()
    filled = inj.df[LEAK_MISSING_COL].notna()
    assert (filled == (df["y"] == rarest)).all()
    assert inj.columns == [LEAK_MISSING_COL]


# ------------------------------------------- truth matches what checks flag ----
def test_duplicates_check_flags_exactly_the_planted_rows():
    inj = inject_duplicates(frame(), "y", np.random.default_rng(0))
    f = next(x for x in duplicates.run(ctx_for(inj.df)) if "exact duplicate" in x.title)
    assert f.evidence["rows"] == inj.rows


@pytest.mark.parametrize("injector", [inject_leak_copy, inject_leak_missingness])
def test_leakage_check_flags_planted_column(injector):
    inj = injector(frame(1000), "y", np.random.default_rng(0))
    flagged = {c for f in leakage.run(ctx_for(inj.df)) for c in f.columns}
    assert set(inj.columns) <= flagged
    assert not {"x1", "x2", "cat"} & flagged  # honest features stay unflagged
