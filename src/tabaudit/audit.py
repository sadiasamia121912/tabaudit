"""Orchestrates loading, task inference and running every registered check."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

from tabaudit import __version__
from tabaudit.checks import REGISTRY
from tabaudit.context import AuditContext
from tabaudit.findings import AuditReport, CheckRun, DatasetSummary, Finding
from tabaudit.loader import dtype_kinds, infer_task, load_table

ProgressFn = Callable[[str, str], None]  # (check_name, status) -> None


def run_audit(
    data: str | Path | pd.DataFrame,
    target: str | None = None,
    test: str | Path | pd.DataFrame | None = None,
    checks: Iterable[str] | None = None,
    max_rows: int = 50_000,
    random_state: int = 42,
    on_progress: ProgressFn | None = None,
) -> AuditReport:
    """Audit a dataset and return an :class:`AuditReport`.

    Parameters
    ----------
    data:    path to CSV/Parquet/… or a DataFrame.
    target:  name of the label column (omit for unsupervised checks only).
    test:    optional held-out set, used to detect train/test contamination.
    checks:  subset of check names to run (default: all).
    """
    df, path = (
        (data, "<DataFrame>") if isinstance(data, pd.DataFrame) else (load_table(data), str(data))
    )
    test_df, test_path = (None, None)
    if test is not None:
        test_df, test_path = (
            (test, "<DataFrame>")
            if isinstance(test, pd.DataFrame)
            else (load_table(test), str(test))
        )

    if target is not None and target not in df.columns:
        close = [c for c in df.columns if c.lower() == target.lower()]
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise KeyError(f"Target column '{target}' not found.{hint}")

    task = infer_task(df[target]) if target else "unsupervised"
    n_classes = (
        int(df[target].nunique(dropna=True)) if target and task == "classification" else None
    )

    ctx = AuditContext(
        df=df,
        target=target,
        task=task,
        test_df=test_df,
        max_rows=max_rows,
        random_state=random_state,
    )
    selected = list(checks) if checks else list(REGISTRY)
    unknown = [c for c in selected if c not in REGISTRY]
    if unknown:
        raise ValueError(f"Unknown check(s): {unknown}. Available: {list(REGISTRY)}")

    findings: list[Finding] = []
    runs: list[CheckRun] = []
    for name in selected:
        fn = REGISTRY[name]
        if on_progress:
            on_progress(name, "running")
        t0 = time.perf_counter()
        try:
            out = fn(ctx)
            findings.extend(out)
            runs.append(CheckRun(name, "ok", time.perf_counter() - t0, len(out)))
        except Exception as exc:
            runs.append(
                CheckRun(name, "error", time.perf_counter() - t0, 0, f"{type(exc).__name__}: {exc}")
            )
        if on_progress:
            on_progress(name, runs[-1].status)

    summary = DatasetSummary(
        path=path,
        n_rows=len(df),
        n_cols=int(df.shape[1]),
        target=target,
        task=task,
        n_classes=n_classes,
        test_path=test_path,
        n_test_rows=len(test_df) if test_df is not None else None,
        memory_mb=round(float(df.memory_usage(deep=True).sum()) / 1e6, 2),
        dtypes=dtype_kinds(df),
    )
    return AuditReport(summary=summary, findings=findings, checks=runs, version=__version__)
