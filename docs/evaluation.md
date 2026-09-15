# How well does tabaudit detect things?

`docs/benchmarks.md` shows what the tool *found* on ten public datasets. That answers "does
it find real problems?" but not "what does it miss, and how often does it cry wolf?" — you
cannot know that on real data, because nobody knows the full truth about a real dataset.

So this page measures the tool on faults **we planted ourselves**. Take a dataset, remove
the problems we already know about, plant a fault whose exact location we know, run the
audit, and compare what it flagged with where we planted. Every number here is
reproducible: `python benchmarks/evaluate.py` (≈ 3 min for all 120 runs) writes
`benchmarks/eval_results.json`; `python benchmarks/sweep_label_noise.py` writes
`benchmarks/sweep_results.json`. Results below are from the committed JSON files, tabaudit
0.1.0 plus the unreleased changes of 2026-09-15 (commits `a8b27de`, `fee377e`).

## Setup

**Datasets.** The same ten as `docs/benchmarks.md` (titanic, adult, credit-g,
telco-customer-churn, bank-marketing, breast-w, heart-statlog, diabetes, spambase,
creditcard), 270 to 275 663 rows after cleaning.

**Clean base.** Before planting anything: drop the leaks we know about (Titanic `boat` and
`body`, bank-marketing `duration`), drop exact duplicates. Then run the audit once on this
base and remember what it flags — the *baseline*. A real dataset has its own label noise,
so some rows are flagged before we touch anything.

**Faults**, one type at a time, three random seeds each:

| fault | what is planted | ground truth |
|---|---|---|
| duplicates | exact copies of 5 % of the rows | the copies' row indices |
| label flips | 3 % of labels changed to a random *other* class | the flipped rows |
| leak (copy) | a new column = target + small Gaussian noise | the column |
| leak (missingness) | a new column of random values, filled in **only** for the rarest class — the Titanic `boat` pattern | the column |

Planted columns have neutral names (`planted_copy`, `planted_missing`), so the leakage
check has to find them from the data, not the name.

**Metrics.**

- **precision** — of the rows we flagged, the share that were really planted;
- **precision excl. baseline** — the same, but ignoring rows the *same rule* already flagged
  on the clean base. Those are the dataset's own noise; counting them as false alarms would
  punish the tool for finding real errors we did not plant;
- **recall** — of the rows we planted, the share we flagged. For large datasets the
  label-noise check works on a 20 000-row sample, so recall is measured against the planted
  rows inside that sample (reproduced through the tool's own sampler, so it cannot drift);
- **FPR** (leakage only) — of the innocent original columns, the share the leakage check
  accused (CRITICAL / HIGH / MEDIUM finding). Identifier-column and INFO findings do not
  count as accusations.

Means are over 10 datasets × 3 seeds = 30 runs per fault.

## Results

| check | fault | detected | precision | precision excl. baseline | recall | FPR |
|---|---|---:|---:|---:|---:|---:|
| duplicates | 5 % copies | 30 / 30 | 1.00 | — | 1.00 | — |
| leakage | target copy + noise | **30 / 30** (all CRITICAL) | — | — | — | **0.00** |
| leakage | filled only for one class | **30 / 30** (all CRITICAL) | — | — | — | **0.00** |
| label noise — *likely* tier | 3 % random flips | — | 0.43 ± 0.23 | **0.80 ± 0.15** | **0.64 ± 0.23** | — |
| label noise — *suspected* tier | 3 % random flips | — | 0.35 ± 0.24 | 0.70 ± 0.18 | 0.73 ± 0.16 | — |

In one sentence: **duplicates and single-column leaks are found every time with no false
accusations; of the rows newly called "likely mislabeled", four in five were planted
errors, and about two in three planted errors were found.**

### Label noise, per dataset (*likely* tier)

The spread behind the ± is not random — it tracks how noisy the dataset already is.

| dataset | rows | flagged before injection | planted flips in sample | precision excl. baseline | recall |
|---|---:|---:|---:|---:|---:|
| creditcard | 275 663 | 14 | 613 | 1.00 | 1.00 |
| spambase | 4 210 | 127 | 126 | 0.90 | 0.87 |
| breast-w | 463 | 16 | 14 | 0.90 | 0.88 |
| adult | 48 790 | 1 103 | 603 | 0.88 | 0.65 |
| bank-marketing | 45 195 | 1 464 | 595 | 0.88 | 0.80 |
| credit-g | 1 000 | 102 | 30 | 0.78 | 0.46 |
| telco-customer-churn | 7 021 | 557 | 211 | 0.78 | 0.49 |
| heart-statlog | 270 | 27 | 8 | 0.70 | 0.42 |
| diabetes | 768 | 75 | 23 | 0.61 | 0.35 |
| titanic | 1 309 | 119 | 39 | 0.57 | 0.47 |

Where the model is accurate and the labels are clean (creditcard, spambase, breast-w) a
planted flip is obvious and almost every one is caught. Where the dataset is itself noisy
and the task hard (diabetes, heart-statlog, titanic: 8–10 % of rows already flagged before
we did anything) a planted flip lands among many genuinely ambiguous rows and both numbers
drop. That is not a tuning problem; it is what "hard dataset" means.

## What the evaluation changed

Building the harness was worth it before it produced a single table, because the first
run found two things wrong with the tool.

**A blind spot in the leakage check.** On the first run the missingness leak was missed on
creditcard, all three seeds. Fraud is 0.17 % of rows, so a 20 000-row sample holds ~36
fraud rows; the single-feature decision tree required at least 40 rows per leaf, so it
*could not* isolate the rows where the planted column was filled in and scored it AUC 0.49
— a perfect leak, invisible. Fix: the leaf floor is capped at half the rarest class, and the
AUC of the bare "is this value missing?" indicator is scored directly. Two regression tests
with a 0.18 % positive class fail on the old code and pass on the new; the ten real-data
scores did not change. Details: `docs/checks.md`, *Rare classes*.

**cleanlab was not helping.** v0.1.0 picked label-noise suspects with the `cleanlab`
library's confident-learning filter and called them "likely" if the model also gave the
label < 20 % probability. `benchmarks/sweep_label_noise.py` computed the out-of-fold
probabilities once per run and scored every rule on the same matrix:

| rule | precision excl. baseline | recall | F1 | rows flagged |
|---|---:|---:|---:|---:|
| self-confidence < 0.3 | 0.70 | 0.73 | **0.71** | 7.9 % |
| self-confidence < 0.2 | 0.80 | 0.64 | **0.70** | 4.9 % |
| cleanlab ∧ self-confidence < 0.2 (v0.1.0 "likely") | 0.79 | 0.57 | 0.66 | 4.3 % |
| cleanlab filter alone (v0.1.0 "suspected") | 0.62 | 0.66 | 0.62 | 8.4 % |

Adding cleanlab's filter to `< 0.2` left precision flat and cost seven points of recall;
`< 0.3` beat the bare filter on every column. Both tiers are now plain self-confidence
thresholds (0.2 / 0.3) and the dependency is gone. `0.3` is the broader tier rather than the
headline because it would have pushed four of the ten real datasets to HIGH at 70 %
precision, while `0.2` changes no dataset's severity. Full table and reasoning:
`docs/checks.md`, *label_noise*.

## What this does **not** show

Be careful quoting these numbers.

- **Planted flips are uniform random; real label noise is not.** A human mislabels the
  ambiguous cases — the borderline tumour, the customer who almost churned — which are
  exactly the rows where the model is least sure and the detector is weakest. Random flips
  hit easy and hard rows alike, so the recall here is an **upper bound** on real-world
  recall. The hand review in `docs/label_noise_review.md` (5 of 10 top suspects clearly
  wrong, 5 ambiguous, 0 false alarms) is the closer-to-reality complement.
- **"Precision excl. baseline" assumes the baseline flags are real noise.** Some of them
  are surely the tool's own mistakes. Raw precision (0.43) and excl-baseline precision
  (0.80) bracket the truth; the truth is nearer the top on clean datasets and nearer the
  bottom on noisy ones.
- **Two leak shapes only.** A column that copies the target, and a column present only
  for one class. Not tested: leaks that need two columns together, leaks through time
  ordering, soft leaks like bank-marketing `duration` (moderately predictive, known only
  after the outcome) — the gap-based MEDIUM rule that catches `duration` on real data is
  not exercised here at all.
- **Duplicates are trivially exact.** A 30/30 on exact copies says the bookkeeping is
  right, nothing more. Near-duplicates (numeric tolerance, whitespace variants) are not
  detected and not tested.
- **Ten datasets, all binary classification.** No regression targets, no multi-class; on
  binary targets `self-confidence < 0.5` already implies the model prefers the other class,
  which is why the `argmax ≠ label` variant in the sweep was identical to the plain rule.
- **Sampled recall is a modelling choice.** On adult, bank-marketing and creditcard only
  planted rows inside the 20 000-row sample count. That is fair to the check as designed,
  but a user who wants every row examined must raise `max_rows`.

## Reproduce

```powershell
python benchmarks/evaluate.py                       # all datasets, 3 seeds, 4 faults
python benchmarks/evaluate.py diabetes --seeds 1    # quick look at one dataset
python benchmarks/sweep_label_noise.py              # the rule/threshold comparison
pip install cleanlab                                # optional: also reproduce the cleanlab rows
```

Both scripts merge into their JSON files, so a partial run never wipes the rest.
