"""Loading datasets and inferring the ML task."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SUPPORTED = {".csv", ".tsv", ".parquet", ".pq", ".feather", ".json"}


def load_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")
    suf = p.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(p, low_memory=False)
    if suf == ".tsv":
        return pd.read_csv(p, sep="\t", low_memory=False)
    if suf in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if suf == ".feather":
        return pd.read_feather(p)
    if suf == ".json":
        return pd.read_json(p, lines=True)
    raise ValueError(f"Unsupported file type '{suf}'. Supported: {sorted(SUPPORTED)}")


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a frame back out in the format its suffix asks for (never with an index column,
    which is what produced half the `Unnamed: 0` findings this tool reports)."""
    p = Path(path)
    suf = p.suffix.lower()
    p.parent.mkdir(parents=True, exist_ok=True)
    if suf == ".csv":
        df.to_csv(p, index=False)
    elif suf == ".tsv":
        df.to_csv(p, sep="	", index=False)
    elif suf in {".parquet", ".pq"}:
        df.to_parquet(p, index=False)
    elif suf == ".feather":
        df.reset_index(drop=True).to_feather(p)
    elif suf == ".json":
        df.to_json(p, orient="records", lines=True)
    else:
        raise ValueError(f"Cannot write '{suf}'. Supported: {sorted(SUPPORTED)}")
    return p


def infer_task(y: pd.Series, max_classes: int = 20) -> str:
    """Classification if the target is categorical/boolean or a low-cardinality integer."""
    if (
        pd.api.types.is_bool_dtype(y)
        or pd.api.types.is_string_dtype(y)
        or y.dtype == object
        or isinstance(y.dtype, pd.CategoricalDtype)
    ):
        return "classification"
    nunique = y.nunique(dropna=True)
    if pd.api.types.is_integer_dtype(y) and nunique <= max_classes:
        return "classification"
    if pd.api.types.is_float_dtype(y):
        # floats that are really integer labels (0.0 / 1.0)
        vals = y.dropna()
        if nunique <= max_classes and np.all(np.equal(np.mod(vals, 1), 0)):
            return "classification"
    return "regression"


def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """Turn any DataFrame into an all-numeric frame suitable for tree models.

    - numeric / bool -> float
    - datetime       -> int64 nanoseconds
    - object/category-> integer codes (NaN preserved)

    The frame is built in one go from the encoded columns: inserting them one at a time
    fragments it, which pandas warns about once per column on wide data (hundreds of
    features, e.g. text embeddings) and which slows everything downstream.
    """
    cols = []
    for i in range(X.shape[1]):
        s = X.iloc[:, i]  # by position, so duplicate column names survive
        if pd.api.types.is_bool_dtype(s):
            cols.append(s.astype(float).to_numpy())
        elif pd.api.types.is_numeric_dtype(s):
            cols.append(pd.to_numeric(s, errors="coerce").astype(float).to_numpy())
        elif pd.api.types.is_datetime64_any_dtype(s):
            cols.append(s.astype("int64").astype(float).where(s.notna(), np.nan).to_numpy())
        else:
            codes, _ = pd.factorize(s, use_na_sentinel=True)
            codes = codes.astype(float)
            codes[codes < 0] = np.nan
            cols.append(codes)
    out = pd.DataFrame(dict(enumerate(cols)), index=X.index)
    out.columns = X.columns
    return out


def dtype_kinds(df: pd.DataFrame) -> dict[str, int]:
    kinds: dict[str, int] = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            k = "bool"
        elif pd.api.types.is_numeric_dtype(s):
            k = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(s):
            k = "datetime"
        else:
            k = "categorical"
        kinds[k] = kinds.get(k, 0) + 1
    return kinds
