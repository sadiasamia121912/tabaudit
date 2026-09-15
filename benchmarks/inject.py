"""Plant known faults in a clean DataFrame so the audit's answers can be scored.

Each injector is a pure, seeded function: it never mutates its input, never touches disk,
and returns the damaged frame together with the *truth* — exactly which rows or column it
planted. `benchmarks/evaluate.py` compares that truth with what `run_audit` flags to get
precision / recall / false-positive rate per check.

    rng = np.random.default_rng(0)
    inj = inject_label_flips(df, "y", rng, frac=0.03)
    inj.df      # the damaged frame
    inj.rows    # index labels of the rows we flipped

Injected columns get names no leakage heuristic reacts to (no "id"/"key"/target name), so
the check has to find them from the data alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LEAK_COPY_COL = "planted_copy"
LEAK_MISSING_COL = "planted_missing"


@dataclass
class Injection:
    kind: str  # "duplicates" | "label_flips" | "leak_copy" | "leak_missingness"
    df: pd.DataFrame
    rows: list = field(default_factory=list)  # index labels of planted rows
    columns: list[str] = field(default_factory=list)  # names of planted columns


def _count(frac: float, n: int) -> int:
    return max(1, round(frac * n))


def inject_duplicates(
    df: pd.DataFrame, target: str, rng: np.random.Generator, frac: float = 0.05
) -> Injection:
    """Append exact copies of `frac` of the rows. Truth = the copies' new index labels,
    which is what `duplicates` flags (keep="first") and what drop_duplicates() removes."""
    k = _count(frac, len(df))
    copies = df.sample(k, random_state=rng.integers(2**31))
    out = pd.concat([df, copies], ignore_index=True)
    return Injection("duplicates", out, rows=list(range(len(df), len(df) + k)))


def inject_label_flips(
    df: pd.DataFrame, target: str, rng: np.random.Generator, frac: float = 0.03
) -> Injection:
    """Change the label of `frac` of the rows to a uniformly random *other* class.
    Classification only. Truth = index labels of the flipped rows."""
    out = df.copy()
    y = out[target]
    classes = y.dropna().unique()
    if len(classes) < 2:
        raise ValueError("need at least two classes to flip labels")
    candidates = y.dropna().index.to_numpy()
    k = _count(frac, len(candidates))
    rows = rng.choice(candidates, size=k, replace=False)
    for r in rows:
        others = classes[classes != y.loc[r]]
        out.loc[r, target] = rng.choice(others)
    return Injection("label_flips", out, rows=sorted(rows.tolist()))


def inject_leak_copy(
    df: pd.DataFrame, target: str, rng: np.random.Generator, noise: float = 0.1
) -> Injection:
    """Add a numeric column that is the target plus a little Gaussian noise — the crudest
    possible leak (think: a "score" written after the outcome was known)."""
    out = df.copy()
    y = out[target]
    codes = y.to_numpy(dtype=float) if pd.api.types.is_numeric_dtype(y) else pd.factorize(y)[0]
    codes = np.asarray(codes, dtype=float)
    scale = np.nanstd(codes) or 1.0
    out[LEAK_COPY_COL] = codes + rng.normal(0.0, noise * scale, len(out))
    return Injection("leak_copy", out, columns=[LEAK_COPY_COL])


def inject_leak_missingness(df: pd.DataFrame, target: str, rng: np.random.Generator) -> Injection:
    """Add a column whose *values* are random noise but which is only filled in for one
    outcome — the Titanic `boat` pattern. Classification: filled for the rarest class.
    Regression: filled where the target is above its median."""
    out = df.copy()
    y = out[target]
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
        mask = (y > y.median()).to_numpy()
    else:
        rarest = y.value_counts().idxmin()
        mask = (y == rarest).to_numpy()
    values = rng.normal(size=len(out))
    out[LEAK_MISSING_COL] = np.where(mask, values, np.nan)
    return Injection("leak_missingness", out, columns=[LEAK_MISSING_COL])


INJECTORS = {
    "duplicates": inject_duplicates,
    "label_flips": inject_label_flips,
    "leak_copy": inject_leak_copy,
    "leak_missingness": inject_leak_missingness,
}
