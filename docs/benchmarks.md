# tabaudit on 9 well-known public datasets

Every dataset below was loaded straight from [OpenML](https://www.openml.org) and audited
with **default settings** — `run_audit(df, target=...)`, no tuning, no column dropping —
exactly what `tabaudit audit data.csv --target y` does. Total wall time for all nine:
about 40 s on a laptop CPU.

Reproduce: `python benchmarks/run_benchmarks.py` (writes `benchmarks/results.json`).
Results as of tabaudit 0.1.0, 2026-09-13.

## Summary

| dataset | rows | cols | score | grade | headline finding |
|---|---:|---:|---:|:-:|---|
| titanic | 1 309 | 14 | 64 | C | **HIGH** `boat` predicts survival on its own (AUC 0.97) — target leakage |
| adult | 48 842 | 15 | 81 | B | 10 rows with identical features but different labels; 52 exact duplicates |
| credit-g | 1 000 | 21 | 93 | A | ~6 % of rows likely mislabeled |
| telco-customer-churn | 7 043 | 20 | 80 | B | `TotalCharges` is numeric but stored as text; 42 conflicting-label rows |
| bank-marketing | 45 211 | 17 | 94 | A | 8 : 1 class imbalance |
| breast-w | 699 | 10 | 67 | C | **HIGH** 236 exact duplicate rows (34 %) · **HIGH** 6 "leaky" features *(false positive — see below)* |
| heart-statlog | 270 | 14 | 93 | A | ~6 % of rows likely mislabeled |
| diabetes | 768 | 9 | 93 | A | ~5 % of rows likely mislabeled |
| spambase | 4 601 | 58 | 75 | B | **HIGH** 391 exact duplicate rows (8.5 %) |
| creditcard | 284 807 | 31 | — | — | not yet run (download repeatedly cut off by OpenML; retry pending) |

**3 of 9 datasets have a CRITICAL/HIGH finding; 9 of 9 have at least one MEDIUM.**
One of the HIGH findings is a false positive the tool should not have raised — details below.

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

**adult / telco — rows with identical features and different labels.** No model can fit
these; they set a hard ceiling on achievable accuracy that nobody notices until they wonder
why the model plateaus.

## Findings that need human judgement

**Label noise (all classification datasets).** The confident-learning check estimates
0.8–6.3 % of rows are likely mislabeled. These are *ranked suspects*, not verdicts — see
the README for measured precision/recall on synthetic noise. The next step in this
benchmark is manually reviewing the top-ranked rows on two or three datasets and recording
whether they hold up.

## What the tool got wrong

Benchmarking against known datasets is exactly how you find a tool's blind spots. Two
were found here and are recorded honestly.

### False positive: breast-w "6 features are suspiciously predictive alone" (HIGH)

Single-feature AUCs on breast-w:

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

**Planned fix:** when several features all clear the threshold with no clear gap, report
an INFO-level "highly separable task" note instead of HIGH-severity leakage, and reserve
HIGH for a feature that stands well above the rest.

### Miss: bank-marketing `duration` (a documented leak)

`duration` is the length of the marketing phone call. It is known only *after* the call
ends — the dataset's own documentation says it "should be discarded if the intention is to
have a realistic predictive model". Its single-feature AUC is **0.805**: the strongest
feature by a wide margin (next best is `month` at 0.65), but below the 0.90 threshold, so
tabaudit said nothing.

This is the same weakness from the other side: an absolute threshold catches near-perfect
leaks and misses "soft" ones. A relative test — *is this feature far above every other
feature?* — would catch `duration` and clear breast-w at the same time.

### Not checked: zeros that mean "missing" (diabetes)

The Pima diabetes dataset is famous for recording impossible zeros (insulin = 0, skin
thickness = 0, blood pressure = 0) where the value was actually unknown. tabaudit has no
zero-as-missing check yet, so it reported nothing. Candidate for a future `schema` rule:
numeric columns with a physically implausible spike at exactly 0.

### Minor: `workclass` flagged for referencing the target `class` (adult, LOW)

The name check does a substring match, so `workclass` matches the target name `class`.
Harmless at LOW severity, but a word-boundary match would avoid it.

## Takeaways

- Out of the box, tabaudit caught the two most-taught leaks/quirks (Titanic `boat`,
  Telco `TotalCharges`) and the well-known duplicate problems in spambase and breast-w.
- Its leakage threshold is calibrated for *near-perfect* leaks. It over-fires on easy
  datasets and under-fires on soft leaks; a relative (gap-based) criterion is the fix.
- Label-noise estimates are consistent (1–6 %) but still need manual verification before
  they can be quoted as fact.
