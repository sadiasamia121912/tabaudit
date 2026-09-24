"""encode_features: every column type becomes float, and wide frames encode without warnings."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from tabaudit.loader import encode_features


def test_wide_frame_encodes_without_fragmentation_warning():
    # 400 numeric columns, as with text embeddings: inserting columns one by one used to
    # raise one pandas PerformanceWarning ("highly fragmented") per column past ~100.
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, 400)), columns=[f"emb_{i}" for i in range(400)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = encode_features(X)
    assert out.shape == X.shape
    assert list(out.columns) == list(X.columns)
    np.testing.assert_array_equal(out.to_numpy(), X.to_numpy())


def test_mixed_types_and_duplicate_names():
    X = pd.DataFrame(
        {
            "flag": [True, False, None],
            "n": [1, 2, 3],
            "when": pd.to_datetime(["2024-01-01", None, "2024-01-03"]),
            "city": ["a", None, "a"],
        },
        index=[10, 11, 12],
    )
    X.insert(4, "n", [4.0, 5.0, 6.0], allow_duplicates=True)
    out = encode_features(X)
    assert list(out.columns) == ["flag", "n", "when", "city", "n"]
    assert all(pd.api.types.is_float_dtype(d) for d in out.dtypes)
    assert list(out.index) == [10, 11, 12]
    assert out.iloc[:, 4].tolist() == [4.0, 5.0, 6.0]
    assert np.isnan(out["when"].iloc[1]) and np.isnan(out["city"].iloc[1])
    assert out["city"].iloc[0] == out["city"].iloc[2]
