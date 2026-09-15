# How each check works — and why the thresholds are what they are

Every number in this document is the one in the code (`src/tabaudit/checks/`). If you
change a threshold, change it here too.

The checks run in a fixed order because later ones depend on earlier ones:

1. **schema** — structure: missing values, constant columns, numbers stored as text
2. **duplicates** — repeated rows, contradictory labels, train/test overlap
3. **imbalance** — class distribution of the target
4. **leakage** — columns that give the answer away
5. **label_noise** — rows whose label looks wrong

`leakage` records which columns it flagged as identifiers or leaks, and `label_noise`
trains *without* them. If it didn't, a leaked column would let the model agree with every
label — including the wrong ones — and the noise would be invisible.

Counting checks (schema, duplicates, imbalance) always run on the full dataset. The two
model-based checks (leakage, label_noise) run on a random sample of at most 50 000 rows
(`--max-rows`), because they cross-validate models and 50 000 rows is where a laptop
starts to notice. Findings say so when a sample was used.

---

## The score

Each finding subtracts a fixed penalty from 100:

| severity | penalty | meaning |
|---|---:|---|
| CRITICAL | 30 | any metric you compute on this data is wrong |
| HIGH | 15 | metrics are probably inflated or misleading |
| MEDIUM | 7 | will bite you; fix before you trust results |
| LOW | 3 | worth a line in the preprocessing script |
| INFO | 0 | context only |

Grades: A ≥ 90, B ≥ 75, C ≥ 60, D ≥ 40, F below.

The penalties were chosen so that the grade matches what a careful reviewer would say:

- **One CRITICAL alone → 70 (C).** A single leak or contaminated test set is enough that
  you should not quote any number from this dataset, but it is usually a one-column fix,
  so it does not go straight to F.
- **Two CRITICALs → 40 (D).** Something is systematically wrong with how the data was
  assembled.
- **One HIGH plus a couple of MEDIUMs → ~70 (C).** Typical of a real, usable dataset with
  known warts (spambase, creditcard).
- **Only LOWs and INFOs → A.** Nothing that changes a conclusion.

Penalties add up rather than taking the maximum so that "many medium problems" is
distinguishable from "one medium problem". The demo dataset (2 CRITICAL, 4 MEDIUM, 2 LOW)
scores 6 — deliberately, so the demo shows what an F looks like.

---

## 1. schema

**What it catches.** The boring problems that break pipelines before any modelling
happens, and that people fix by hand in every tutorial without noticing they are data
defects.

**How.**

- *Mostly-missing columns* — MEDIUM when a column is **≥ 50 %** empty. Above half, any
  imputation is inventing more values than were measured; the column is usually either
  dead or only filled in for a special sub-population (which is a leakage smell — see
  Titanic `body`).
- *Missing target values* — HIGH, any count. Unlabelled rows either crash the training
  step or, worse, get silently dropped and bias the class balance.
- *Constant columns* — LOW. Zero information; some libraries also divide by zero when
  scaling them.
- *Near-constant columns* — LOW when one value covers **≥ 99 %** of rows. Below 99 % the
  minority value can still be a meaningful rare flag, so the check stays out of the way.
- *Numbers stored as text* — LOW when **≥ 95 %** of a text column's non-empty values
  (checked on the first 2 000) parse as numbers after removing thousands separators. The
  5 % slack is for the stray `"?"`, `"N/A"` or blank that caused pandas to read the column
  as text in the first place (Telco `TotalCharges` has 11 blanks in 7 043 rows).
- *Leftover index columns* — INFO for anything named `Unnamed: …`, the signature of
  `to_csv()` without `index=False`.

**What it misses.** Values that are *semantically* missing but stored as a number — the
zeros in the Pima diabetes dataset (insulin = 0 meaning "not measured"). A spike at
exactly zero in a column that cannot physically be zero is on the roadmap.

---

## 2. duplicates

**What it catches.** Rows that appear more than once, rows that contradict each other,
and test rows the model has already seen.

**How.** Every row is hashed with `pandas.util.hash_pandas_object`, so all three checks
are one pass over the data regardless of column count.

- *Exact duplicates* (all columns identical) — severity scales with the fraction:
  **≥ 5 % HIGH, ≥ 1 % MEDIUM, otherwise LOW.** The harm is that a duplicated row lands in
  both the training fold and the validation fold, so cross-validation partly measures
  memorisation. At 1 % the effect on a CV score is already visible; at 5 % (spambase is
  8.5 %) it can move a leaderboard.
- *Conflicting labels* — MEDIUM when rows with identical *features* have different
  *targets*. No model can fit both, so these rows set a hard ceiling on accuracy that
  nobody notices until the model plateaus. It is one severity level regardless of count
  because even a handful indicates a labelling-process problem worth understanding.
- *Train/test overlap* — rows in the test set whose features also appear in training:
  **≥ 1 % of the test set CRITICAL, otherwise HIGH.** This is the most direct way to get a
  test score that means nothing. Even one overlapping row is HIGH because it is never an
  accident of the data — it is a splitting bug, and where there is one there are more.

**What it misses.** Near-duplicates (same row with a rounding difference, a trailing space,
or a different timestamp). Exact hashing is deliberate: it makes the check O(n) and never
produces a false positive. Fuzzy matching is on the roadmap.

---

## 3. imbalance

**What it catches.** Class distributions that make accuracy meaningless and that break
stratified cross-validation.

**How.** Ratio = size of the largest class ÷ size of the smallest.

- **≥ 100 : 1 HIGH** — the majority-class baseline is above 99 %; almost every default
  (accuracy, unweighted loss, random splits) is wrong. Credit-card fraud is 578 : 1.
- **≥ 10 : 1 MEDIUM** — baseline above 90 %; class weights and PR-AUC needed, but standard
  tooling copes.
- **≥ 3 : 1 LOW** — worth reporting balanced metrics; below 3 : 1 the check says nothing,
  because a 70 / 30 split is normal and flagging it would be noise.
- *Rare classes* — MEDIUM for any class with **fewer than 10 examples**. Five-fold
  stratified CV needs at least five examples per class just to put one in each fold, and
  anything under ten cannot be learned or evaluated; merge or collect more.
- *One class only* — CRITICAL. It is not a classification dataset.

The recommendation text is deliberately specific about **not oversampling before
splitting**: doing so copies minority rows into the validation fold and is the most common
way an imbalance "fix" creates a leak.

**What it misses.** Imbalance in a *regression* target (a long tail of rare high values).
Regression targets are skipped entirely.

---

## 4. leakage

**What it catches.** Columns that make a model look excellent in evaluation and useless in
production, because they contain information that is only available *after* the outcome
is known.

**How — three independent tests.**

*a) Identifier-like columns* — MEDIUM. A non-float column that is **≥ 98 % unique**, or
≥ 90 % unique with an ID-ish name (`id`, `key`, `code`, `ref`, `number`, …). Trees
memorise row identity from IDs, and sequential IDs leak collection order. Floats are
exempt because a continuous measurement is naturally unique.

*b) Single-feature predictive power* — the main test. For every feature *on its own*, a
shallow decision tree (depth 4, minimum leaf size n / 500, but never more than half the
rarest class — see *Rare classes* below) is 5-fold cross-validated
against the target and scored with AUC (classification) or R² (regression). Then:

| rule | severity | why |
|---|---|---|
| score **≥ 0.98** | CRITICAL | No honest single feature predicts a real-world target this well. This is the answer written down after the fact. |
| score **≥ 0.90** *and* stands alone | HIGH | Strong, and nothing else comes close — the signature of a leak (Titanic `boat`: 0.97 vs next-best 0.74). |
| score **≥ 0.75** *and* stands alone | MEDIUM "soft leak" | Not dramatic, but a lone dominant feature is often something recorded *during* the outcome (bank-marketing `duration`: 0.81 vs 0.65). |
| score ≥ 0.90 but part of a crowd | INFO "highly separable task" | Several strong features bunched together mean the task is easy, not leaky (breast-w: six features between 0.90 and 0.97). |

"Stands alone" means: sort all features by score; a group of at most **2** features at
the top whose drop to the next-best feature is **≥ 0.15** AUC. Why those numbers:

- **0.15** is 30 % of the usable AUC range (0.5 = chance, 1.0 = perfect). On the
  benchmark datasets it separates the two documented leaks from every honest feature with
  room on both sides:

  | dataset | best feature | next best | gap | |
  |---|---|---|---:|---|
  | titanic | `boat` 0.967 | `sex` 0.742 | **0.225** | leak |
  | bank-marketing | `duration` 0.805 | `month` 0.649 | **0.155** | leak |
  | diabetes | `plas` 0.771 | `age` 0.685 | 0.085 | honest |
  | credit-g | `checking_status` 0.680 | `duration` 0.618 | 0.062 | honest |
  | spambase | `char_freq_!` 0.823 | `capital_run_length_longest` 0.806 | 0.017 | honest |
  | heart-statlog | `chest` 0.741 | `thal` 0.728 | 0.013 | honest |
  | adult | `relationship` 0.778 | `marital-status` 0.768 | 0.009 | honest |
  | breast-w | `Cell_Size_Uniformity` 0.970 | `Cell_Shape_Uniformity` 0.967 | 0.003 | honest |
  | telco-customer-churn | `Contract` 0.735 | `tenure` 0.733 | 0.002 | honest |

  The largest honest gap (0.085) and the smallest leak gap (0.155) are almost a factor of
  two apart, and 0.15 sits in between. It is still one threshold fitted on ten datasets;
  a leak that shares its signal with an honest feature will slip under it.
- **at most 2** because a leak is one column (occasionally two that encode the same
  thing, like Titanic `boat` and `body`). Five strong features are an easy dataset.
- The **0.98 / 0.90 / 0.75** absolute lines come from the same benchmark: no honest
  feature reached 0.98; honest features above 0.90 only occurred in crowds; and 0.75 is
  low enough to catch `duration` but high enough that the "stands alone" test, not the
  absolute score, does the work.

Why a tree rather than correlation: correlation only sees linear, numeric relationships.
A depth-4 tree handles categorical codes, thresholds and "any non-missing value means
yes" patterns, while being too small to memorise a high-cardinality column. Why
cross-validated: an in-sample tree would fit noise and every feature would look
predictive.

*Missingness.* The commonest real leak is a field that is only filled in for one class
(`churn_reason`, Titanic `boat`). Two things catch it. Before scoring, missing values are
replaced by a sentinel below the column minimum so the tree can split on "is this
missing?" itself. Independently, for every column with any missing values, the AUC of the
bare indicator *is this value missing?* against the target is computed, and the column's
score is the **higher** of the two. The finding says so explicitly when missingness alone
has AUC ≥ 0.9.

*Rare classes.* The leaf-size floor of n / 500 exists so a noisy feature cannot look
predictive by memorising a handful of rows. But it must not exceed half the rarest class:
on `creditcard` a 20 000-row sample holds ~36 fraud rows, and with the default leaf of 40 the
tree could not isolate them, so a column filled in *only* for fraud scored AUC 0.49 — a
perfect leak, invisible. The fault-injection evaluation (`docs/evaluation.md`) found this;
both the cap and the direct missingness score were added in response (after the v0.1.0
release; they ship with the next version).

*c) Column names that mention the target* — LOW. Whole-word match, so `class_of_service`
matches target `class` but `workclass` does not. It is only a hint; derived columns are
usually caught by (b) anyway.

**What it misses.** Leaks that need *two or more* columns together (a ratio of two honest
features that equals the target); leaks through *time* (a feature computed with data from
after the prediction moment, but not strongly predictive on its own); and group leakage
(the same customer in train and test under different row values). Group/time leakage is
the first roadmap item.

---

## 5. label_noise

**What it catches.** Rows whose label is probably wrong — a ranked list of suspects, not a
verdict.

**How.** Self-confidence of an out-of-fold model:

1. Train a gradient-boosting classifier with 5-fold cross-validation and keep the
   **out-of-fold** predicted probabilities for every row — each row is predicted by a
   model that never saw it, so the model cannot simply have memorised the given label.
2. For each row take the probability the model gives the row's *given* label — its
   **self-confidence**. A low value means the model, having learned the pattern from the
   other rows, confidently disagrees with the label.
3. Two tiers, both plain thresholds: **likely** mislabeled when self-confidence is
   **< 20 %**, **suspected** when **< 30 %**. Suspects are ranked lowest-confidence first.

The model is deliberately regularised (200 rounds at learning-rate 0.05, at most 15
leaves, minimum 40 rows per leaf, L2 = 1, early stopping). An over-confident model calls
its own mistakes label errors.

Severity is by the *likely* fraction: **≥ 8 % HIGH, ≥ 3 % MEDIUM, ≥ 0.5 % LOW, else
INFO.**

**Why these rules and thresholds — and why not cleanlab.** v0.1.0 used the `cleanlab`
library's *confident learning* filter (Northcutt et al., 2021) to pick suspects, and
"likely" was that filter *and* self-confidence < 20 %. The fault-injection benchmark
(`benchmarks/sweep_label_noise.py`, `docs/evaluation.md`) planted 3 % random label flips in
each of 10 public datasets × 3 seeds, computed the out-of-fold probabilities once, and scored
every rule on the same matrix. "Precision excl. baseline" ignores rows the same rule already
flagged *before* injection — those are the dataset's own noise, not false alarms.

| rule | precision (raw) | precision excl. baseline | recall | F1 | rows flagged | note |
|---|---:|---:|---:|---:|---:|---|
| `sc<0.3` | 0.35 | 0.70 | 0.73 | 0.71 | 7.9 % | **chosen: suspected** |
| `sc<0.2` | 0.42 | 0.80 | 0.64 | 0.70 | 4.9 % | **chosen: likely** |
| `sc<0.4` | 0.29 | 0.61 | 0.81 | 0.68 | 11.8 % |  |
| `cl & sc<0.3` | 0.42 | 0.71 | 0.63 | 0.66 | 5.9 % |  |
| `cl & sc<0.2` | 0.45 | 0.79 | 0.57 | 0.66 | 4.3 % | v0.1.0 likely |
| `cl & sc<0.4` | 0.39 | 0.65 | 0.65 | 0.64 | 7.2 % |  |
| `cl` | 0.37 | 0.62 | 0.66 | 0.62 | 8.4 % | v0.1.0 suspected |
| `sc<0.1` | 0.56 | 0.84 | 0.45 | 0.56 | 2.3 % |  |
| `cl & sc<0.1` | 0.56 | 0.84 | 0.42 | 0.53 | 2.2 % |  |

Two things fell out. Adding cleanlab's filter on top of `sc<0.2` left precision unchanged
(0.79 vs 0.80) and cost 7 points of recall: its per-class threshold is the class's *mean*
self-confidence, which on an easy task is ~0.94, so a row the model is 85 % sure is
mislabeled does not count as "confident enough". And `sc<0.3` beat the bare cleanlab filter
on precision, recall, F1 *and* rows flagged. So both tiers became self-confidence thresholds
and the dependency was removed. `0.3` is not the headline tier because it would have pushed
four of the ten datasets (titanic, credit-g, heart-statlog, diabetes) to HIGH at 70 %
precision; `0.2` changes no dataset's severity. (`argmax ≠ label ∧ sc<t` was also tested and
is identical to `sc<t` on binary targets, since `sc < 0.5` already implies the model prefers
the other class.)

**How to read the number.** Of the rows called "likely", about four in five of the *newly*
flagged ones were real planted errors, and about two in three of the planted errors were
found — so the count is an *estimate*, and the value is in the ranked list. A manual review of the top suspects on
two public datasets (`label_noise_review.md`) found 5 of 10 clearly wrong, 5 ambiguous,
none obviously fine.

**What it misses.** Label errors on rows the model is genuinely unsure about (a wrong
label near the decision boundary looks like a correct one); regression targets (no
noise check yet — residual-based detection is on the roadmap); and anything when a class
has fewer than 5 examples, since 5-fold stratified CV is impossible.
