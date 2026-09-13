"""The object every check receives."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from tabaudit.loader import encode_features


@dataclass
class AuditContext:
    df: pd.DataFrame
    target: str | None
    task: str  # classification | regression | unsupervised
    test_df: pd.DataFrame | None = None
    max_rows: int = 50_000  # sample cap for expensive checks
    random_state: int = 42
    # Columns earlier checks decided must not be used as model inputs (IDs, leaky columns).
    excluded_features: set[str] = field(default_factory=set)
    _encoded: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def feature_cols(self) -> list[str]:
        return [c for c in self.df.columns if c != self.target]

    @property
    def X(self) -> pd.DataFrame:
        return self.df[self.feature_cols]

    @property
    def y(self) -> pd.Series | None:
        return self.df[self.target] if self.target else None

    @property
    def X_encoded(self) -> pd.DataFrame:
        if self._encoded is None:
            self._encoded = encode_features(self.X)
        return self._encoded

    @property
    def X_encoded_clean(self) -> pd.DataFrame:
        """Encoded features minus anything flagged as an identifier or leak."""
        keep = [c for c in self.X_encoded.columns if c not in self.excluded_features]
        return self.X_encoded[keep]

    def sample_index(self) -> pd.Index:
        """Row index to use for model-based checks (capped for speed)."""
        if len(self.df) <= self.max_rows:
            return self.df.index
        return self.df.sample(self.max_rows, random_state=self.random_state).index
