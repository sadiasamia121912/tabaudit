# `tabaudit fix` — what it changes, and what it refuses to

`tabaudit audit` tells you what is wrong. `tabaudit fix` repairs the subset of that which has
**exactly one defensible answer**, and — the part that matters — refuses the rest out loud.

```bash
tabaudit fix train.csv --target churn --test test.csv
tabaudit fix train.csv --target churn --drop-leaky --flag-noise --out clean.csv
```

It never edits the input. It writes `<name>.clean.csv` (or `--out`) plus
`<name>.fixplan.json`, a machine-readable record of every fix applied, every fix skipped and
why. The exit code is **0 even when fixes were skipped**: skipping something you did not
authorise is correct behaviour, not a failure.

## The rule

| finding | fixed? | why |
|---|---|---|
| Exact duplicate rows | ✅ automatic | dropping the later copies is what `drop_duplicates()` does |
| Rows shared between train and test | ✅ automatic | dropped **from train only** — shrinking the test set would silently change what any later score means |
| Constant columns | ✅ automatic | a single-valued column carries zero information, by definition |
| Leftover `Unnamed: 0` index columns | ✅ automatic | an export artefact, never data |
| Numbers stored as text | ✅ automatic | `"1,234"` → `1234.0` recovers the column's real type |
| Rows with no label | ✅ automatic | you cannot train on an unlabeled row |
| Target leakage / identifier-like columns | ⚠️ `--drop-leaky` | only you know when a column becomes available. Auto-dropping would delete real features |
| Likely-mislabeled rows | ⚠️ `--flag-noise` | adds a boolean `tabaudit_suspect` column. Never relabels, never drops |
| Conflicting labels on identical rows | ❌ never | which of the two labels is right is not in the data |
| ≥ 50 % missing columns, near-constant columns | ❌ never | drop-or-impute is a modelling decision; the rare value may be the whole point |
| Class imbalance | ❌ never | resampling vs. class weights vs. a different metric is a model choice — and resampling *before* the split is itself leakage |
| Scaling, encoding, imputation | ❌ **never written to a file** | see below |

## Why there is no "cleaned and normalized" CSV

A scaler or an encoder has to be fitted on the training fold and applied to the others. If
`fix` normalised the file, the mean and variance of the *test* rows would be baked into the
numbers the model trains on — which is precisely the contamination `tabaudit audit` exists to
detect. A tool that fixed leakage by introducing leakage would be worse than no tool.

So transformations are emitted as **code you run inside a pipeline**, not applied to data.
(`tabaudit fix` handles the row/column edits above; pipeline generation is the next piece of
this phase.)

## Worked example: the fix that changes nothing

`bank-marketing` (45 211 rows). The audit scores it 83 B and finds two things: `duration`
stands far above every other feature, and ~5.6 % of rows look mislabeled.

```
$ tabaudit fix bank.csv --target Class

Health score before fixing  83/100 B
     finding                                 action         result
 ?   1 feature(s) stand far above all        drop_columns   needs --drop-leaky
     others - possible soft leak             V12
     leakage
 ?   ~2,544 rows (5.6%) are likely           flag_rows      needs --flag-noise
     mislabeled, 656 more suspected
     label_noise
unchanged: 45,211 rows, 17 columns — nothing here has exactly one right answer
2 fix(es) need your say-so: --drop-leaky --flag-noise
```

Nothing was changed, and that is the correct output. `duration` is the length of the sales
call — it is a leak *if* you predict before calling and a legitimate feature *if* you predict
after, and no amount of statistics settles which. An "auto-clean" tool would have deleted the
column (destroying a real feature for anyone doing post-call scoring) or kept it silently
(leaving the leak in place). This one hands the decision back, with the evidence attached:
the `impact` check has already measured that `duration` is worth 0.133 AUC (0.933 → 0.800),
so you know exactly what the decision costs before you make it.

## Worked example: the fix that changes a lot

The demo dataset has planted defects, so the safe fixes have real work to do:

```
$ tabaudit fix tabaudit_demo/churn_train.csv --target churn --test tabaudit_demo/churn_test.csv

 ?   1 feature(s) predict the target almost perfectly   drop_columns   needs --drop-leaky
 ?   1 identifier-like column(s) present as features    drop_columns   needs --drop-leaky
 ✔   1 constant column(s)                               drop_columns   dropped 1 column(s)
 ✔   105 test rows (8.2%) also appear in the training…  drop_rows      dropped 106 row(s)
 ✔   160 exact duplicate rows (3.2%)                    drop_rows      dropped 158 row(s)
 ?   ~378 rows (7.6%) are likely mislabeled             flag_rows      needs --flag-noise
4,992 → 4,728 rows, 11 → 10 columns
```

Auditing the train file on its own, the score goes **36 F → 46 D** on the safe fixes alone,
and **36 F → 87 B** once you add `--drop-leaky --flag-noise`. The gap between 46 and 87 is
the part the tool will not take on itself.

Two details visible in that transcript:

- **160 duplicates, 158 dropped.** The cross-split fix ran first and had already removed two
  of them. Order matters: columns are dropped first, then dtypes coerced, then rows dropped,
  then rows flagged — so the flag column can never be dropped or coerced by a later step.
- **105 test rows, 106 train rows dropped.** The overlap is counted on the test side and
  repaired on the train side, and one train row was duplicated.

## Guards

- The **target column is never dropped**, even if a fix names it.
- A `drop_columns` fix that would leave **no feature columns** is skipped with that reason.
- **Applying a plan twice is a no-op** — rows and columns already gone are reported as
  "already applied" rather than failing.
- The input file is never written to.

## Known wrinkle

`--flag-noise` adds a `tabaudit_suspect` column. On a dataset where fewer than 1 % of rows
are suspects, a later `tabaudit audit` of the cleaned file will flag that column as
near-constant — the tool complaining about its own output. It is cosmetic (a LOW finding),
but it is real, and worth knowing before you see it.
