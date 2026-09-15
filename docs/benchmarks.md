# tabaudit on 10 well-known public datasets

Every dataset below was loaded straight from [OpenML](https://www.openml.org) and audited
with **default settings** — `run_audit(df, target=...)`, no tuning, no column dropping —
exactly what `tabaudit audit data.csv --target y` does. Total wall time: about 2 min on a
laptop CPU for the nine small datasets, plus ~3.5 min for `creditcard` (285k rows).

Reproduce: `python benchmarks/run_benchmarks.py` (writes `benchmarks/results.json`).
Results as of tabaudit 0.1.0 with the gap-based leakage rule, 2026-09-14. Re-run 2026-09-15
after the rare-class leakage fix (`docs/checks.md`, *Rare classes*): every score and every
CRITICAL/HIGH/MEDIUM finding is unchanged; the only difference is creditcard's INFO note,
which now lists 5 strong features instead of 3 because the tree can finally isolate the fraud rows.

## Summary

| dataset | rows | cols | score | grade | headline finding |
|---|---:|---:|---:|:-:|---|
| titanic | 1 309 | 14 | 64 | C | **HIGH** `boat` predicts survival on its own (AUC 0.97, next-best 0.74) — target leakage |
| adult | 48 842 | 15 | 84 | B | 5 groups of rows with identical features but different labels; 52 exact duplicates |
| credit-g | 1 000 | 21 | 93 | A | ~6 % of rows likely mislabeled |
| telco-customer-churn | 7 043 | 20 | 80 | B | `TotalCharges` is numeric but stored as text; 18 conflicting-label groups |
| bank-marketing | 45 211 | 17 | 83 | B | **MEDIUM** `duration` stands far above every other feature (AUC 0.81 vs 0.65) — a documented leak |
| breast-w | 699 | 10 | 82 | B | **HIGH** 236 exact duplicate rows (34 %) · INFO: 6 features ≥ 0.90 alone, "highly separable task" |
| heart-statlog | 270 | 14 | 93 | A | ~6 % of rows likely mislabeled |
| diabetes | 768 | 9 | 93 | A | ~5 % of rows likely mislabeled |
| spambase | 4 601 | 58 | 75 | B | **HIGH** 391 exact duplicate rows (8.5 %) |
| creditcard | 284 807 | 30 | 78 | B | **HIGH** 578 : 1 class imbalance · 9 144 exact duplicate rows (3.2 %) |

**4 of 10 datasets have a CRITICAL/HIGH finding; 10 of 10 have at least one MEDIUM.**
Every HIGH finding is a real, documented property of the dataset (the one false positive
from the first run — breast-w's "leaky" features — was fixed; see "What the tool got wrong").

## Findings that are definitely real

**Titanic — `boat` is target leakage.** `boat` is the lifeboat number. Only passengers who
made it into a lifeboat have one, so *whether the field is blank* predicts survival with
AUC 0.97. The check flagged it specifically through the missingness path. Anyone who trains
on Titanic with `boat` left in gets a ~97 % model that has learned nothing about
survival. (`body` — body-recovery number, only for the dead — is 91 % missing and was
flagged by the schema check rather than the leakage check; its single-feature AUC is
lower because most of the dead were never recovered.)

**Telco churn — `TotalCharges` stored as text.** Eleven rows contain a blank string
instead of a number, so pandas reads the whole column as text. Every tutorial on this
dataset has a line fixing exactly this.

**spambase — 391 exact duplicates (8.5 %).** A known property of the dataset. With
duplicates present, the same row lands in both a training and a validation fold and
cross-validation scores are inflated.

**breast-w — 236 exact duplicates (34 %).** OpenML's copy has the patient-ID column
removed; with only nine integer features on a 1–10 scale, many rows coincide exactly. The
rows *are* identical, so the CV-contamination problem is real, but the number overstates
"true" duplicates.

**creditcard — 578 : 1 imbalance and 9 144 duplicates.** Fraud is 0.17 % of transactions,
so accuracy is meaningless (predicting "not fraud" scores 99.8 %) — the dataset's defining
property. The duplicate count is higher than the 1 081 usually quoted because OpenML's copy
drops the `Time` column; the remaining 29 features coincide far more often. As with
breast-w, the tool is right about the data it was given.

**bank-marketing — `duration` is a soft leak (MEDIUM).** Call duration is only known after
the call, and the dataset's own documentation says to drop it for realistic modelling. Its
single-feature AUC is a modest 0.81, but the next-best feature is at 0.65 — a gap of 0.16
that no honest feature in the other nine datasets shows. Caught by the relative rule
described below.

**adult / telco — rows with identical features and different labels.** No model can fit
these; they set a hard ceiling on achievable accuracy that nobody notices until they wonder
why the model plateaus.

## Findings that need human judgement

**Label noise (all classification datasets).** The label-noise check estimates
0.9–6.3 % of rows are likely mislabeled. These are *ranked suspects*, not verdicts — see
the README for measured precision/recall on synthetic noise. The next step in this
benchmark was to review the top-ranked rows by eye: `python benchmarks/show_suspects.py
heart-statlog` prints each suspect next to a "typical" row of each class. **Result
([full sheet](label_noise_review.md)): of the 10 top suspects on heart-statlog and
diabetes, 5 look genuinely mislabeled, 5 are ambiguous, 0 look like false alarms.** Every
"mislabeled" case has all its key clinical markers on the opposite side from its label.

## What the tool got wrong

Benchmarking against known datasets is exactly how you find a tool's blind spots. Two
were found on the first run, and fixing them is what produced the numbers above.

### Fixed — false positive: breast-w "6 features are suspiciously predictive alone" (HIGH)

On the first run (absolute AUC ≥ 0.90 rule) breast-w scored 67 / C with a HIGH leakage
finding. Single-feature AUCs on breast-w:

| feature | AUC alone |
|---|---:|
| Cell_Size_Uniformity | 0.970 |
| Cell_Shape_Uniformity | 0.967 |
| Bland_Chromatin | 0.932 |
| Bare_Nuclei | 0.931 |
| Single_Epi_Cell_Size | 0.909 |
| Clump_Thickness | 0.900 |

None of these are leaks. Cell-size and cell-shape measurements genuinely predict
malignancy; the task is simply easy. The leakage check uses an *absolute* threshold
(AUC ≥ 0.90 → suspicious), which cannot distinguish "one column secretly encodes the
answer" from "every column is strongly informative".

Compare Titanic, where the real leak stands **alone**: `boat` at 0.97, next best
(`sex`) at 0.74. A leak is an *outlier*; an easy dataset is a *crowd*.

**The fix (now in `checks/leakage.py`):** sort the single-feature scores and find the
largest drop between neighbours. If one or two features sit ≥ 0.15 above everything else,
they stand alone → HIGH (≥ 0.90) or MEDIUM "soft leak" (≥ 0.75). Strong features bunched
together are reported at INFO as a "highly separable task". Near-perfect features
(≥ 0.98) stay CRITICAL regardless. After the fix breast-w scores 82 / B with the duplicate
finding as its only HIGH, and Titanic's `boat` is unchanged.

### Fixed — miss: bank-marketing `duration` (a documented leak)

`duration` is the length of the marketing phone call. It is known only *after* the call
ends — the dataset's own documentation says it "should be discarded if the intention is to
have a realistic predictive model". Its single-feature AUC is **0.805**: the strongest
feature by a wide margin (next best is `month` at 0.65), but below the old 0.90 threshold,
so the first run said nothing.

The same weakness from the other side: an absolute threshold catches near-perfect leaks and
misses "soft" ones. The relative rule above catches `duration` (gap 0.16) as a MEDIUM
soft leak — and, checked across all ten datasets, flags nothing else, so the rule is not
just trading one false positive for another.

### Not checked: zeros that mean "missing" (diabetes)

The Pima diabetes dataset is famous for recording impossible zeros (insulin = 0, skin
thickness = 0, blood pressure = 0) where the value was actually unknown. tabaudit has no
zero-as-missing check yet, so it reported nothing. Candidate for a future `schema` rule:
numeric columns with a physically implausible spike at exactly 0.

### Fixed — minor: `workclass` flagged for referencing the target `class` (adult, LOW)

The name check did a substring match, so `workclass` matched the target name `class`.
It now matches whole words (`class_of_service` yes, `workclass` no).

## Takeaways

- Out of the box, tabaudit caught the three most-taught leaks/quirks (Titanic `boat`,
  bank-marketing `duration`, Telco `TotalCharges`), the defining imbalance of creditcard,
  and the well-known duplicate problems in spambase and breast-w.
- The first run's absolute leakage threshold over-fired on easy datasets and under-fired
  on soft leaks. Replacing it with a relative (gap-based) rule fixed both cases without
  introducing a new false positive on any of the ten datasets.
- Label-noise estimates are consistent (1–6 %). A manual review of the top 10 suspects
  found 5 that look clearly wrong and none that look like false alarms — the *ranking* is
  trustworthy, even though the exact percentage should not be quoted as fact.
