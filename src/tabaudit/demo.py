"""Synthetic customer-churn dataset with deliberately planted defects.

Used by `tabaudit demo` so anyone can see every check fire in ten seconds.
Planted problems (all of which tabaudit should catch):

  * customer_id          - identifier column left in as a feature
  * churn_reason         - target leakage: only filled in for churned customers
  * ~4% duplicate rows
  * ~6% flipped labels   - label noise
  * ~15% churn rate      - moderate class imbalance
  * referral_code        - 80% missing
  * region_code          - constant column
  * test split shares ~3% of rows with train - contamination
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_churn_dataset(
    n: int = 6000,
    seed: int = 7,
    noise_rate: float = 0.06,
    dup_rate: float = 0.04,
    return_truth: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return (train, test). With ``return_truth=True`` also return a boolean Series,
    aligned to ``train``, marking rows whose label was flipped."""
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 72, n)
    monthly = np.round(rng.normal(65, 20, n).clip(15, 150), 2)
    support_calls = rng.poisson(1.2, n)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    payment = rng.choice(["card", "bank", "cheque", "wallet"], n)
    age = rng.integers(18, 80, n)

    # True churn propensity from real signal.
    logit = (
        -4.6
        - 0.07 * tenure
        + 0.035 * monthly
        + 0.9 * support_calls
        + np.where(contract == "month-to-month", 2.0, np.where(contract == "one-year", 0.3, -1.2))
    )
    p = 1 / (1 + np.exp(-logit))
    churn_true = (rng.random(n) < p).astype(int)

    # Label noise: what the database *recorded*, which is what we will train on.
    churn = churn_true.copy()
    flip = rng.random(n) < noise_rate
    churn[flip] = 1 - churn[flip]

    # Leaky column: filled in by the retention team only after churn was recorded.
    reasons = rng.choice(["price", "service", "competitor", "moved"], n)
    churn_reason = np.where(churn == 1, reasons, None)

    # 80%-missing column, constant column.
    referral = np.where(rng.random(n) < 0.2, rng.integers(1000, 9999, n).astype(str), None)
    region_code = np.full(n, "EU-W")

    df = pd.DataFrame(
        {
            "customer_id": [f"C{100000 + i}" for i in range(n)],
            "age": age,
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "support_calls": support_calls,
            "contract": contract,
            "payment_method": payment,
            "referral_code": referral,
            "region_code": region_code,
            "churn_reason": churn_reason,
            "churn": churn,
            "_flipped": flip,
        }
    )

    # Duplicates.
    dups = df.sample(int(n * dup_rate), random_state=seed)
    df = (
        pd.concat([df, dups], ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    # Train/test split with contamination.
    test = df.sample(int(len(df) * 0.2), random_state=seed + 1)
    train = df.drop(test.index)
    leak = train.sample(int(len(test) * 0.03), random_state=seed + 2)
    test = pd.concat([test, leak]).sample(frac=1, random_state=seed + 3).reset_index(drop=True)
    train = train.reset_index(drop=True)
    flipped = train.pop("_flipped").astype(bool)
    test = test.drop(columns="_flipped")
    if return_truth:
        return train, test, flipped
    return train, test


def write_demo(out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, test = make_churn_dataset()
    tp, sp = out / "churn_train.csv", out / "churn_test.csv"
    train.to_csv(tp, index=False)
    test.to_csv(sp, index=False)
    return tp, sp
