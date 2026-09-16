# Changelog

All notable changes to tabaudit. Versions follow [Semantic Versioning](https://semver.org).

## Unreleased

### Added
- **`impact` check** — prices what `leakage` flagged instead of only naming it: two
  cross-validated models on the same rows and folds, one with every feature and one without
  the flagged columns, reported as both held-out scores and the gap (AUC, or R² for
  regression). titanic: **AUC 0.991 with `boat` + `name`, 0.875 without**. Always INFO
  (penalty 0), so pricing a defect never scores it twice and no benchmark score moved; it
  runs only when something was flagged, so a clean dataset pays no runtime for it (`adult`:
  0.00 s). Where it does run it fits two models: 1.1 s on titanic, 13.6 s on bank-marketing's
  45 k rows, and it obeys `--max-rows`.
- **`tabaudit fix`** — applies the fixes that have exactly one defensible answer (exact
  duplicates; rows shared with the test set, dropped from *train* only; constant and leftover
  `Unnamed:` columns; numbers stored as text; rows with no label) and reports every other one
  as skipped. Target leakage and identifier columns need `--drop-leaky`; likely-mislabeled rows
  need `--flag-noise`, which adds a boolean `tabaudit_suspect` column and never relabels or
  drops. Scaling, encoding and imputation are never written to a file — that is how test
  statistics leak into training data. Writes `<name>.clean.csv` + `<name>.fixplan.json`, leaves
  the input untouched, and exits 0 with fixes outstanding. On `bank-marketing` the correct
  output is that nothing changes; see [`docs/fix.md`](docs/fix.md).
- **Generated pipeline code** — every `tabaudit fix` run also writes `<name>_pipeline.py`
  (`--no-pipeline` to skip): a `ColumnTransformer` + `Pipeline` from the cleaned frame's
  dtypes — `StandardScaler` for numeric, `OneHotEncoder(handle_unknown="ignore")` up to 20
  levels, `OrdinalEncoder` above that, `SimpleImputer` only where nulls exist, datetimes left
  out with a note, flagged-but-kept columns in `EXCLUDED` with the finding that named them,
  and `remainder="drop"`. It runs as written; a test executes it end-to-end.
- **`Fix` on every finding that has one** — `action` / `params` / `safe` / `flag`, serialised
  with the finding, plus `tabaudit.fix.apply_fixes(df, report, flags)` and a `FixPlan` that
  records what was applied, what was skipped and why.
- `tabaudit.loader.write_table` — writes CSV / TSV / Parquet / Feather / JSON-lines back out,
  never with an index column.
- `benchmarks/run_benchmarks.py` now also applies the safe fixes and re-audits, recording
  rows/columns removed and the score after. `docs/benchmarks.md` gains a table of it: five of
  ten datasets improve (breast-w 82 B → 97 A, spambase 75 B → 90 A), five are untouched, and
  none scores worse — which the runner asserts on every run.
- **Per-check score breakdown** — `AuditReport.score_breakdown` (also in the JSON report,
  as "by check" bars in the terminal report and in the HTML health card): the same 0–100
  scale per check, so it is visible where the points went. A breakdown, not an average.

## 0.2.0 — 2026-09-15

The evaluation release: tabaudit now measures itself, and two things it found were fixed.

### Added
- **Fault-injection evaluation** (`benchmarks/evaluate.py`, `benchmarks/inject.py`,
  `docs/evaluation.md`): plants known duplicates, label flips and leaky columns in the 10
  benchmark datasets and reports precision / recall / false-positive rate per check.
  Duplicates 1.00 / 1.00; both planted leak shapes detected 30/30 with 0 innocent columns
  accused; label noise ("likely") precision 0.80 excluding pre-existing noise, recall 0.64.
- **`tabaudit gate FILE...`** — audit several files, one pass/fail line each, exit 1 if any
  fails (2 if a file cannot be read). `--fail-on none` gates on the score alone.
- **`--fail-on SEVERITY`** on `audit` and `gate`: exit 1 on any finding at that severity or
  worse. Combines with the existing `--fail-under`.
- **GitHub Action** (`action.yml`, use as `sadiasamia121912/tabaudit@v0.2.0`) and
  **pre-commit hook** (`.pre-commit-hooks.yaml`) wrapping the gate. CI runs the Action
  against the demo data and asserts it fails on the planted leak.
- Findings now list every affected row: `evidence["rows"]` on the exact-duplicates and
  label-noise findings, plus `evidence["rows_likely"]` for label noise.
- README: "Use it as a gate", "How well does it detect things?", "How it compares"
  (vs ydata-profiling, deepchecks, cleanlab — install size and time measured).

### Changed
- **Label noise no longer uses cleanlab.** Both tiers are thresholds on the out-of-fold
  model's self-confidence: *likely* < 0.2, *suspected* < 0.3. The fault-injection sweep
  (`benchmarks/sweep_label_noise.py`) showed cleanlab's confident-learning filter left
  precision unchanged and cost 7 points of recall on top of self-confidence. On the 10
  benchmark datasets every score and severity is unchanged. `cleanlab` is removed from the
  dependencies.

### Fixed
- **Leakage check was blind to leaks confined to a rare class.** The single-feature tree
  required `n // 500` rows per leaf; when the minority class in the sample was smaller than
  that (creditcard: 36 fraud rows vs a leaf of 40) a column filled in only for that class
  scored AUC 0.49. The leaf is now capped at half the rarest class, and the bare
  "is this value missing?" indicator is scored directly. Found by the evaluation harness;
  regression tests with a 0.18 % positive class.

### Internal
- `benchmarks/` is linted in CI; `tests/` can import the benchmark scripts.
- Windows/Linux CI matrix unchanged (Python 3.10 / 3.12 / 3.13).

## 0.1.0 — 2026-09-14

First release: five checks (schema, duplicates, imbalance, leakage, label_noise), 0–100
health score, terminal / HTML / JSON reports, `--fail-under`, `tabaudit demo`, benchmark
results on 10 public datasets (`docs/benchmarks.md`).
