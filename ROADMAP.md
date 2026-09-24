# tabaudit — Roadmap

_Last updated: 2026-09-17. Companion to `../AI_ML_Portfolio_Projects.md` (the 3-project plan)._

## Where things stand

| Item | Status |
|------|--------|
| Core tool (5 checks, CLI, HTML/JSON report, `demo`) | ✅ v0.1.0 done |
| Tests (`pytest`) | ✅ 29 passing |
| Lint (`ruff`) | ✅ clean |
| GitHub Actions CI (Linux/Windows × Py 3.10/3.12/3.13) | ✅ configured |
| Pushed to GitHub (`sadiasamia121912/tabaudit`) | ✅ |
| Benchmarks on real datasets | ✅ 10 datasets in `docs/benchmarks.md`, gap-based leakage fix verified on them |
| README benchmark table + GIF | ✅ |
| Public repo | ✅ 2026-09-17 |
| PyPI release (`pip install tabaudit`) | ✅ **0.3.1 live 2026-09-25** (0.3.0 on 09-16, 0.2.0 on 09-15, 0.1.0 on 09-14), verified in a fresh venv |

**How to get running again (every session):**
```powershell
cd C:\Users\User\dev\tabaudit\tabaudit
.\.venv\Scripts\activate
pytest -q                      # should say "29 passed"
tabaudit demo                  # see every check fire on synthetic data
```

---

## Phase 1 — Real-world evidence  (2 days)  ✅ done 2026-09-14

Goal: a table of *real* findings on datasets everyone recognises. This is the number
that goes on the résumé.

- [x] **1.1** Write `benchmarks/run_benchmarks.py`
  - loads each dataset via `sklearn.datasets.fetch_openml(name, data_home="benchmarks/openml_cache")`
  - calls `tabaudit.run_audit(df, target=...)` on each
  - writes one row per dataset to `benchmarks/results.json` (score, grade, counts per severity, key findings)
  - Already cached (from last session): `titanic`, `adult`, `credit-g`, `creditcard`, `telco-customer-churn`
- [x] **1.2** Added 5 more: `heart-disease` / `breast-w` (Breast Cancer Wisconsin), `bank-marketing`, `house_prices` (regression — tests the R² path)
- [x] **1.3** Run it. Expect a few minutes for `creditcard` (285k rows → sampled to 50k). _(creditcard ran 2026-09-14: 218 s)_
- [x] **1.4** Write `docs/benchmarks.md`: one table (dataset · rows · score · grade · headline finding) + a paragraph per interesting result
- [x] **1.4b** Fix the two leakage blind spots the benchmark exposed (breast-w false positive, bank-marketing `duration` miss) with a gap-based rule; re-run all 10 datasets. _(done 2026-09-14)_
- [x] **1.5** Manually verify 2–3 flagged label-noise rows _(done 2026-09-14 → `docs/label_noise_review.md`: 5 of 10 look mislabeled, 5 ambiguous, 0 false alarms)_:
  ```powershell
  python benchmarks/show_suspects.py heart-statlog --top 5 --out docs/label_noise_review.md
  python benchmarks/show_suspects.py diabetes --top 5 --out docs/label_noise_review.md
  ```
  Each suspect is printed next to a "typical" row of each class. For each one, fill in the
  `verdict` (looks mislabeled / plausible / can't tell) and `why` columns of the review
  sheet. Then write 2–3 sentences of conclusion at the top of `docs/label_noise_review.md`
  ("N of M top suspects looked genuinely wrong") and link it from `docs/benchmarks.md`.
  This is what makes the claim credible in an interview.
- [x] **1.6** Commit (`benchmarks/openml_cache/` is git-ignored).

**Done when:** `docs/benchmarks.md` has ≥ 8 datasets and you can say "found X in N of 8". _(Currently: real CRITICAL/HIGH findings in 4 of 10, at least one MEDIUM in 10 of 10.)_

## Phase 2 — Polish  (1 day)  ✅

- [x] **2.1** README: paste the benchmark table under a new "Results on real datasets" section _(2026-09-14)_
- [x] **2.2** Record a ~20 s terminal GIF of `tabaudit demo` → `docs/demo.gif`, embed in README _(2026-09-14; rebuild with `powershell docs/make_demo_gif.ps1`)_
- [x] **2.3** `docs/checks.md`: how each check works, every threshold and why, what each misses; plus how the score is computed _(2026-09-14)_
- [x] **2.4** Tick the "Audit results for popular public benchmark datasets" box in the README roadmap _(2026-09-14)_
- [x] **2.5** _(done 2026-09-17)_ Make the repo **public**

## Phase 3 — Publish  (½ day)  ← 3.5 still open (LinkedIn post)

- [x] **3.1** Create a PyPI account + API token (https://pypi.org) _(done 2026-09-14; tokens live in `C:\Users\User\.pypirc`)_
- [x] **3.2** ~~Dry run on TestPyPI~~ _skipped: `twine check` + fresh-venv install of the local wheel covered it; TestPyPI needs a separate account and wasn't worth the friction_
  ```powershell
  python -m build
  twine upload --repository testpypi dist/*
  pip install -i https://test.pypi.org/simple/ tabaudit   # in a fresh venv
  ```
- [x] **3.3** Real upload _(2026-09-14, https://pypi.org/project/tabaudit/0.1.0/)_: `twine upload dist/*`, then verify `pip install tabaudit && tabaudit --version` in a **fresh** venv
- [x] **3.4** `git tag v0.1.0`; GitHub release with notes + wheel/sdist attached _(2026-09-14)_
- [ ] **3.5** LinkedIn/blog post: *"I audited 8 popular ML datasets — here's what's wrong with them"* (the benchmark table + 3 concrete examples)
- [x] **3.6** Fill in the numbers in the résumé bullet in `../AI_ML_Portfolio_Projects.md` _(2026-09-14)_

**Done when:** `pip install tabaudit` works on a clean machine and the repo is public with a release.

## Phase 4 — Prove it: fault-injection evaluation  (1 day)  ✅ done 2026-09-15

Goal: today the evidence is "found real problems in 4 of 10 datasets". An interviewer will
ask *"how do you know it isn't missing things, and what's the false-positive rate?"* — and
right now there is no answer. This phase produces one: **precision / recall per check**,
measured the way the cleanlab and deepchecks papers measure themselves.

**Technique — fault injection:** start from a dataset the tool already knows well, *plant*
faults whose location you know exactly, run the audit, and compare what it flagged against
where you planted them.

| Check | What we inject | Ground truth | Metric |
|---|---|---|---|
| duplicates | copy 5 % of rows | indices of the copies | recall (should be 1.0 — a sanity check, the hash is exact) |
| label noise | flip 3 % of labels to a random *other* class | indices flipped | precision + recall, separately for the "likely" and "suspected" tiers |
| leakage | (a) `target + noise` column, (b) a column whose *missingness* mirrors the target (like Titanic `boat`) | the injected column name | detection rate; plus **FPR** = original, non-leaky columns that get flagged |

Definitions (one line each): *precision* = of the rows we flagged, what fraction were really
planted; *recall* = of the rows we planted, what fraction we flagged; *FPR* = of the innocent
columns, what fraction we wrongly accused.

- [x] **4.1** _(done 2026-09-15: `evidence["rows"]` on both findings, plus `rows_likely` for label noise; 30 tests pass)_ Findings must expose *which* rows they mean, or nothing can be scored. Add
  `"rows": [...]` (all affected indices, not just the top 25) to the `evidence` of the
  label-noise finding (both tiers) and the exact-duplicates finding. Leakage already has
  `columns`. Keep the existing top-25 `suspects` list for the report. Tests: `rows` has the
  expected length on `demo` data; JSON report still serialises.
- [x] **4.2** _(done 2026-09-15: 4 injectors + 12 tests, incl. "truth == what the check flags" for duplicates and both leaks; 42 tests pass)_ `benchmarks/inject.py` — one function per fault, each `(df, target, rng) ->
  (df_injected, truth)`: `inject_duplicates(frac=0.05)`, `inject_label_flips(frac=0.03)`,
  `inject_leak_copy(noise=0.1)`, `inject_leak_missingness()`. Pure functions, seeded, no I/O.
  Tests (`tests/test_inject.py`): each produces exactly the promised count and `truth` points
  at the right rows/column.
- [x] **4.3** _(done 2026-09-15: 120 runs in ~2 min; results in `benchmarks/eval_results.json`; smoke test in `tests/test_evaluate.py`)_ `benchmarks/evaluate.py` — for each of the 10 datasets × 3 seeds: build a
  *clean base* first (drop the already-known leaks — Titanic `boat`/`body`, bank-marketing
  `duration` — and `drop_duplicates()`), then inject **one** fault type at a time, run
  `run_audit`, score against `truth`. Write `benchmarks/eval_results.json`. Cap `max_rows`
  at 20 000 for this run (label-noise CV is the bottleneck; `creditcard` alone was 218 s).
  Print a table: check · precision · recall · FPR, mean ± sd over seeds.
- [x] **4.4** _(done 2026-09-15: `benchmarks/sweep_label_noise.py`; both tiers are now self-confidence thresholds 0.2 / 0.3; cleanlab dependency removed; real-data scores unchanged)_ Calibrate, don't guess. `LIKELY_MAX_SELF_CONFIDENCE = 0.2` in
  `checks/label_noise.py` is a made-up number — sweep {0.1, 0.2, 0.3, 0.4} against the
  injected truth and keep the value with the best F1 on the "likely" tier. Do the same for
  the leakage gap rule *only if* FPR > 5 %. Change a constant only when the data says so, and
  record the sweep table in `docs/checks.md` next to the threshold.
  **Early evidence (from the 4.1 test, 2026-09-15):** on a perfectly separable toy set with
  20 planted flips, the model gives the planted labels only 1–18 % probability, yet cleanlab's
  `confident_learning` filter flags just 7–11 of them. Reason: it only counts a disagreement
  when the other class's probability beats that class's *threshold* = mean self-confidence of
  the class (≈ 0.94 on an easy task), so a row at 88 % "you're wrong" is not confident
  *enough*. `filter_by="predicted_neq_given"` and the plain rule `self_conf < 0.2` both got
  19–20/20 with 0 false alarms on the toy. So the sweep should also compare *filters*, not just
  the 0.2 constant: `confident_learning ∧ self_conf<t` (today) vs `self_conf<t` alone vs
  `predicted_neq_given ∧ self_conf<t`. Judge on real datasets — the toy has no ambiguous rows,
  real data does (`docs/label_noise_review.md`).
  **First full run (4.3, 10 datasets × 3 seeds):** duplicates 1.00 / 1.00; `leak_copy`
  30/30 detected, FPR 0.00; `leak_missingness` **27/30**, FPR 0.00; label noise "likely" tier
  precision 0.43 raw → **0.81 excluding rows already flagged before injection**, recall 0.53;
  "suspected" tier 0.37 → 0.63, recall 0.65. Per-dataset spread is large: spambase / creditcard
  / breast-w ≥ 0.85 excl-baseline precision, credit-g / diabetes ≈ 0.3 (they are genuinely
  noisy: 13 % of rows flagged *before* injection).
  - [x] **4.4a — fix first, then calibrate.** _(done 2026-09-15: leaf cap + missingness score; 2 regression tests; `leak_missingness` 30/30; real-data benchmark scores unchanged)_ The 3 misses are all `creditcard`: in a 20 k
    sample there are 36 fraud rows but the single-feature tree in `checks/leakage.py` uses
    `min_samples_leaf = max(5, n // 500) = 40`, so it can never isolate the rows where the
    planted column is filled → AUC 0.49. The existing `_missingness_auc` helper returns 1.0 on
    the same column but is only used to decorate finding text. Fix: cap `min_samples_leaf` at
    half the minority-class count, **and** promote missingness-AUC to a real detection path
    (a column whose *missingness alone* scores ≥ NEAR_PERFECT is a leak). Add a test with a
    0.2 % positive class. Re-run `evaluate.py creditcard` → expect 3/3.
- [x] **4.5** _(done 2026-09-15)_ `docs/evaluation.md`: the per-check table, then an honest "what this does not
  show" paragraph — injected flips are *uniform random*; real label noise is
  class-conditional and feature-dependent (see `docs/label_noise_review.md`), so the recall
  number is an **upper bound**. Link it from README under a new "How well does it detect
  things?" section, and update the résumé bullet in `../AI_ML_Portfolio_Projects.md`.
- [x] **4.6** _(each step was committed and pushed as it finished)_ Commit, push. (`benchmarks/eval_results.json` is committed — it *is* the
  evidence; the OpenML cache stays ignored.)

**Done when:** `docs/evaluation.md` has precision/recall/FPR for every check over ≥ 8 datasets
× 3 seeds, and you can say a sentence like *"label-noise detection: 0.9x precision / 0.8x
recall on 3 % injected noise; leakage: 10/10 detected, x % false-positive rate."*

## Phase 5 — Adoption: make it a tool people run  (½ day, optional)

Goal: change the story from "a script that prints a report" to "a data-quality gate in your
CI pipeline". Each item is small; the sum is what makes it look like a real tool.

- [x] **5.1** _(done 2026-09-15; `--fail-under` already existed since v0.1.0 — the roadmap had mis-named it `--fail-below` — so this added `--fail-on`)_ `tabaudit audit data.csv --target y --fail-under 70` → exit code 1 when the
  score is under the bar; `--fail-on high` → exit 1 on any finding of that severity or worse.
  Exit codes are how CI systems decide pass/fail. Tests for both flags.
- [x] **5.2** _(done 2026-09-15; added `tabaudit gate FILE...` so one invocation handles many files, which pre-commit needs; CI has an `action-self-test` job that runs the Action on demo data and asserts it fails on the planted leak. **README snippets pin `@v0.2.0` — that tag must exist before the repo goes public.**)_ `action.yml` (composite GitHub Action: install tabaudit, run with `--fail-under`)
  and `.pre-commit-hooks.yaml`. A 6-line usage example of each in the README.
- [x] **5.3** _(done 2026-09-15; measured in a throwaway venv, since deleted)_ README "How it compares": one table vs `ydata-profiling`, `deepchecks`,
  `cleanlab` — which of the 5 checks each covers, gives a score?, fixes?, install size,
  wall time on `adult`. Say where tabaudit loses. Interviewers trust a project that names its
  competitors.
- [ ] ~~**5.4**~~ _skipped 2026-09-15: creditcard already runs in ~20 s and adult in ~10 s; nothing worth an hour_ Profile the `creditcard` run (`python -m cProfile -s cumtime …`). The
  suspect is the 5-fold `cross_val_predict` in label noise. Cap that check's sample at 20 k or
  lower `max_iter` on large samples; target < 60 s. Skip if it isn't a one-hour fix.

## Phase 6 — v0.3.0: `tabaudit fix`  (1–2 days, optional)

_Renumbered from v0.2.0: the evaluation + gate work shipped as 0.2.0 on 2026-09-15._

Goal: tabaudit currently *detects and scores*. v0.2 makes it *fix what has exactly one
correct fix*, refuse to guess on the rest, and hand the user leak-free preprocessing code.
That last sentence is the pitch — the "auto-clean everything" version would be wrong half the
time (see `docs/label_noise_review.md`: 5 of 10 suspects were ambiguous) and wrong silently.

**Design principle (decided 2026-09-15):**

| Finding | Auto-fix? | Why |
|---|---|---|
| Exact duplicates | ✅ safe | drop, keep first (cross-split dupes: drop from *train*, never test) |
| Constant / ID-like columns, mixed dtypes | ✅ safe | drop / coerce |
| Class imbalance | ❌ | resample vs. class weights vs. change metric is a model decision; resampling before the split is itself leakage |
| Leakage | ❌ | only a human knows if `duration` is recorded after the outcome; auto-dropping would kill real features |
| Label noise | ❌ | flag rows, never relabel or drop |
| Normalization / scaling / encoding | ❌ never applied to the file | must be fit on train *inside* the pipeline; a "normalized CSV" bakes test statistics into training |

- [x] **6.1** _(done 2026-09-16)_ `Fix` dataclass in `findings.py`: `action` (`drop_rows` | `drop_columns` | `flag_rows` | `coerce_dtype`), `params: dict`, `safe: bool`, `flag: str | None` (the CLI flag that enables an unsafe fix). Add optional `fix: Fix | None = None` to `Finding`. Include it in `to_dict()`. Tests: serialisation round-trip.
- [x] **6.2** _(done 2026-09-16)_ Emit fixes from the checks that can — `duplicates.py` (drop_rows, safe), `schema.py` (drop_columns / coerce_dtype, safe), `leakage.py` (drop_columns, **unsafe**, flag `--drop-leaky`), `label_noise.py` (flag_rows, unsafe, flag `--flag-noise` → adds a `tabaudit_suspect` bool column). `imbalance.py` emits **no** fix — its recommendation text is the fix. Tests: each check's fix has the right rows/columns on `demo` data.
- [x] **6.3** _(done 2026-09-16)_ `fix.py`: `apply_fixes(df, report, enabled_flags) -> (clean_df, FixPlan)`. Apply order matters: drop columns first, then drop rows, then flag rows. `FixPlan` records what was applied, what was skipped and why, row/col counts before/after. Tests: applying the plan twice is a no-op; skipped unsafe fixes are listed.
- [x] **6.4** _(done 2026-09-16)_ `tabaudit fix data.csv --target y [--drop-leaky] [--flag-noise] [--out clean.csv]`. Prints the plan (✔ applied / ? needs a flag), writes `<name>.clean.csv` + `<name>.fixplan.json`. Exit code 0 even when unsafe fixes are skipped — skipping is the correct behaviour, not an error.
- [x] **6.5** _(done 2026-09-16)_ `pipeline.py`: generate `<name>_pipeline.py` — a scikit-learn `ColumnTransformer` skeleton from the cleaned frame's dtypes: `StandardScaler` for numeric, `OneHotEncoder(handle_unknown="ignore")` for categoricals with ≤ 20 levels, `OrdinalEncoder` above that, `SimpleImputer` where nulls were found. Header comment explaining *why this is code and not a transformed CSV* (fit on train only). This is generated **text**, not applied transformation — keep it that way.
- [x] **6.6** _(done 2026-09-16)_ Re-run `benchmarks/run_benchmarks.py` with `fix` on the 10 datasets → add a "rows/cols removed by safe fixes" column to `docs/benchmarks.md`. Sanity check: score after `fix` ≥ score before on every dataset.
- [x] **6.7** _(done 2026-09-16)_ `docs/fix.md`: the table above + one worked example (bank-marketing `duration`). README section "Fixing what it finds". Bump to 0.3.0, `python -m build`, `twine upload`, tag, release.
- [~] **6.8** _(drafted 2026-09-16 → `../linkedin_post_tabaudit_v0.3.md`; **not posted** — both posts wait on the repo going public, since their GitHub links 404 until then)_ Second LinkedIn post: *"v0.3: tabaudit now fixes what it finds — and why it refuses to fix some things"*.

- [x] **6.9** _(done 2026-09-16)_ **`impact` check — what the flagged columns are worth.** Two cross-validated models on the same rows and folds (`X_encoded` vs `X_encoded_clean`, the regularised GBM `label_noise` already uses): held-out AUC/R² with the leaky + identifier columns and without, reported as both scores plus the gap. Severity **INFO / penalty 0** on purpose — the defect is already scored by the check that flagged it, so no benchmark score moved. Runs only when something was flagged (`adult`: 0.00 s, no finding); 1.1 s on titanic, 13.6 s on bank-marketing's 45 k rows (two model fits, obeys `--max-rows`). Real results: **titanic 0.991 → 0.875 (gap 0.116, `boat` + `name`)**, bank-marketing 0.933 → 0.800 (gap 0.133, the documented `duration` leak), demo 1.000 → 0.761 (gap 0.239). Score invariance verified on real data too: titanic still 64 C, bank-marketing still 83 B, exactly as in `docs/benchmarks.md`. 6 new tests, incl. one that locks the score-invariance and one that documents the honest caveat (an *honest* dominant feature is priced identically — the gap is the size of the bet, not proof).
- [x] **6.10** _(done 2026-09-16)_ **Per-check score breakdown.** `AuditReport.score_breakdown` (registry order, `status == "ok"` only), in `to_dict()`, as "by check" bars in the terminal verdict panel (worst first) and in the HTML health card + a `score` column in the checks table. Display only — no weights re-derived, so the score is unchanged; documented in `docs/checks.md` as a breakdown, *not* an average.
- [x] **6.11** _(done 2026-09-17)_ Rebuild `docs/demo.gif` before the 0.3.0 release — the verdict panel now carries the per-check bars, so the recorded GIF is out of date (`powershell docs/make_demo_gif.ps1`).

**Done when:** `tabaudit fix` on `tabaudit demo` data drops the duplicates and constant column, leaves the leaky column in place with a clear message, and the generated pipeline file runs end-to-end on the clean CSV.

## Phase 7 — Stretch (only if Phase 6 is finished)

- [ ] Group / time leakage check: entity IDs that appear in both train and test; date columns where test dates precede train dates
- [ ] Near-duplicate detection (numeric tolerance / fuzzy text)
- [ ] Label-noise for regression targets (residual-based)
- [ ] `pre-commit` hook / reusable GitHub Action
- [ ] Score validation study (from `../update.md`): inject faults at increasing severity and
  check whether the health score predicts the gap between a model's *reported* and its true
  held-out performance. The one genuinely research-shaped idea in that document, and it reuses
  `benchmarks/evaluate.py` wholesale. After Project 2, not before — it is 2–3 days.
- **Declined, on the record:** drift / outlier / model-evaluation checks (`../update.md` idea 6).
  That is deepchecks' and Evidently's ground, and the README's "How it compares" now states the
  narrow scope as a position rather than a gap. Adding them makes tabaudit a worse deepchecks.
  Also declined: task-aware check selection (idea 5 — the task is already inferred, the rest is
  advice text) and per-column counterfactual datasets (idea 4 — 80 % of it is 6.9 at several
  times the cost).

## Then → Project 2 (LLM → tiny model distillation)

Do **not** start until Phase 3 is complete. Recommended order after that (decided 2026-09-15):
**Phase 4 (prove it) → Phase 5 (adoption) → Phase 6 (`fix`) → Project 2**, about a week total.
Phases 5–6 are optional — decide after Phase 4 whether they or Project 2 are the better use of the next week. See `../AI_ML_Portfolio_Projects.md`.

---

## Rules of thumb

- One commit per finished checkbox; push at the end of each session.
- Run `pytest -q && ruff check src tests examples benchmarks` before every commit.
- If something is taking > 2× the estimate, cut scope, don't extend time.

---

## Session log

**2026-09-13** — Phase 1 tasks 1.1–1.4 done, pushed (`f7b68aa`). `.venv` and dataset cache
deleted to free space. **To resume:**
```powershell
cd C:\Users\User\dev\tabaudit\tabaudit
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
pytest -q                                   # expect 22 passed
python benchmarks/run_benchmarks.py         # re-downloads datasets (~1 min), ~40 s to audit
```
Next: task 1.5 (manually verify label-noise suspects), retry `creditcard` on a good
connection, then Phase 2 — starting with the gap-based leakage fix from `docs/benchmarks.md`.

**2026-09-14** — Rebuilt `.venv`. Gap-based leakage rule implemented (`split_stand_alone` in
`checks/leakage.py`, 7 new tests), `creditcard` finally ran, `docs/benchmarks.md` rewritten
for 10 datasets, `benchmarks/show_suspects.py` added for task 1.5. **To resume:** same
commands as above. Task 1.5 done: `docs/label_noise_review.md` has verdicts + reasoning for 10 suspects (written with Claude's help — read it before an interview; the key idea is that row numbers mean nothing, feature values decide). **Phase 1 complete. Next: Phase 2**, starting with 2.1 (paste the benchmark table into the README).

**2026-09-14 (later)** — Phase 2 done except 2.5 (make public — your click). Phase 3
started: `dist/` builds clean (`twine check` PASSED), wheel verified in a fresh venv,
`.pypirc` created. `twine upload --repository testpypi dist/*` returned **403 Forbidden** —
not retried. **To resume:**
```powershell
cd C:\Users\User\dev\tabaudit\tabaudit
.\.venv\Scripts\activate          # if .venv was deleted: python -m venv .venv; pip install -e ".[dev]"
pytest -q                          # expect 29 passed
```
Then fix the 403 before anything else — check, in this order: (1) tokens not swapped
between `[pypi]` and `[testpypi]` in `C:\Users\User\.pypirc` (the two accounts are
separate); (2) TestPyPI account email verified; (3) token pasted whole, `pypi-` prefix
included, no trailing space. Then `python -m build && twine upload --repository testpypi dist/*`
and continue with 3.2's fresh-venv install check, 3.3, 3.4.

**2026-09-14 (evening)** — **`tabaudit 0.1.0` is on PyPI** and verified with a clean
`pip install tabaudit`; tag `v0.1.0` + GitHub release created; résumé bullet filled in.
TestPyPI skipped (see 3.2). Two tokens were accidentally printed in the session and were
revoked and replaced. **Left:** 2.5 make repo public (your click — do this before sharing the
PyPI link, since the README links point at the repo) and 3.5 the LinkedIn/blog post. Then
**Project 2**. Phase 4 stretch items stay parked.

**2026-09-15** — No code changes. Decided the design for v0.2 `tabaudit fix` (safe fixes
auto-applied, unsafe ones behind flags, preprocessing emitted as sklearn code — never as a
transformed CSV) and wrote it up as Phase 4. Still open before that: 2.5 (make repo public)
and 3.5 (LinkedIn post).

**2026-09-15 (later)** — Still no code. Agreed the improvement strategy: credibility >
features. Added **Phase 4 — fault-injection evaluation** (precision/recall/FPR per check)
and **Phase 5 — adoption** (CI exit codes, GitHub Action, comparison table, profiling);
the `fix` design moved to **Phase 6**, stretch to **Phase 7**. Next: 2.5 and 3.5, then 4.1.

**2026-09-15 (evening)** — **4.1 done.** `duplicates` and `label_noise` findings now list every
affected row in `evidence["rows"]` (`rows_likely` too for label noise); verified in the JSON
report on demo data. Found while writing the test: cleanlab's `confident_learning` recovers
only ~half of obvious planted flips — noted under 4.4, *not* changed yet. **Next: 4.2**
(`benchmarks/inject.py`). Resume: activate `.venv`, `pytest -q` → 30 passed.

**2026-09-15 (night)** — **4.2 done.** `benchmarks/inject.py`: `inject_duplicates`,
`inject_label_flips`, `inject_leak_copy`, `inject_leak_missingness`, each returning an
`Injection(kind, df, rows, columns)`. Tests import it via `pythonpath = ["benchmarks"]` in
`pyproject.toml`; CI now lints `benchmarks/` too. **Next: 4.3** (`benchmarks/evaluate.py`).
Resume: activate `.venv`, `pytest -q` → 42 passed.

**2026-09-15 (late)** — **4.3 done.** `benchmarks/evaluate.py` runs 10 datasets × 3 seeds ×
4 faults in ~2 min (only the checks relevant to each fault are run). Numbers are in the 4.4
notes above and in `benchmarks/eval_results.json`. Two things it exposed: (1) leakage goes blind
when the minority class in the sample is smaller than the tree's `min_samples_leaf`
(creditcard, 3/3 misses) — precise cause and fix written up as **4.4a**; (2) raw label-noise
precision is dominated by each dataset's *pre-existing* noise, so `precision_excl_baseline`
is the number to quote. **Next: 4.4a** (fix), then 4.4 (threshold/filter sweep).
Resume: activate `.venv`, `pytest -q` → 44 passed; `python benchmarks/evaluate.py creditcard`.

**2026-09-15 (night, later)** — **4.4a done.** `checks/leakage.py`: `min_samples_leaf` is
capped at half the rarest class, and the missingness-only AUC is now a real score (the column
keeps the higher of tree score and missingness score). Two regression tests with a 0.18 %
positive class fail on the old code and pass on the new. `evaluate.py`: `leak_missingness`
27/30 → **30/30**, FPR still 0.00, nothing else moved. `run_benchmarks.py` re-run: all 10
scores identical; only creditcard's INFO list grew from 3 to 5 strong features.
`docs/checks.md` and `docs/benchmarks.md` updated. **Next: 4.4** (label-noise threshold and
filter sweep). Resume: activate `.venv`, `pytest -q` → 46 passed.

**2026-09-15 (late night)** — **4.4 done.** `benchmarks/sweep_label_noise.py` computes the
out-of-fold probabilities once per (dataset, seed) and scores 13 rule × threshold variants on
the same matrix. Result: cleanlab's confident-learning filter added nothing over plain
self-confidence (same precision, −7 recall), and `sc<0.3` beat bare cleanlab on every metric.
**Decision (yours): remove cleanlab.** `label_noise.py` now: likely = self-conf < 0.2,
suspected = < 0.3; dependency dropped; verified tests + demo pass with cleanlab uninstalled.
Planted-flip numbers (per-tier baseline): likely 0.79/0.57 → **0.80/0.64**, suspected
0.62/0.66 → **0.70/0.73**. All 10 real-data scores and severities unchanged. Docs, README table
(re-measured), résumé wording updated. `evaluate.py` now uses per-tier baselines. **Next: 4.5**
(`docs/evaluation.md`, README section, résumé numbers). Resume: activate `.venv`
(`pip install -e ".[dev]"` if rebuilt — cleanlab no longer needed), `pytest -q` → 46 passed.

**2026-09-15 (end of day)** — **4.5 done → Phase 4 complete.** `docs/evaluation.md` written
(setup, results table, per-dataset label-noise table, what the harness changed, what it does
*not* show, reproduce). README gained "How well does it detect things?"; résumé bullets and
interview talking points updated in `../AI_ML_Portfolio_Projects.md`. **Still open: 2.5 (make
repo public) and 3.5 (LinkedIn post) — both yours.** Then decide: release the unreleased
changes as 0.1.1 now, or fold into 0.2.0 with Phase 6; and whether Phase 5 (adoption) is worth
the ½ day. Resume: activate `.venv`, `pytest -q` → 46 passed.

**2026-09-15 (Phase 5 started)** — 5.1 `--fail-on SEVERITY` (`--fail-under` already existed).
5.2: `tabaudit gate FILE...` (one line per file, exit 1 on any failure, 2 on unreadable file,
`--fail-on none` for score-only gating), `action.yml` composite Action that installs the
pinned ref itself (not PyPI, so flags always match the code), `.pre-commit-hooks.yaml`, CI
self-test of the Action, README "Use it as a gate". 51 tests. The Action snippet pins
`@v0.2.0` → **a release must come before the repo goes public**. **Next: 5.3** (comparison
table) or skip to the release.

**2026-09-15 (5.3)** — README "How it compares" vs ydata-profiling 4.18, deepchecks 0.19,
cleanlab 2.9: coverage matrix, install size on top of the shared 314 MB stack (14 / 338 / 176 /
2–124 MB), time on `adult` (9.8 / 18.8 / 7.9 / 10–99 s), and an honest "where tabaudit loses".
Notable: deepchecks 0.19.1 would not import without `numpy<2` and `scikit-learn<1.8`;
ydata-profiling pins `pandas<3` and prints a deprecation notice. Scratch venv deleted.
**Next: 5.4** (profiling — probably skip: creditcard already runs in ~20 s) or the release.

**2026-09-15 (release prep)** — Version bumped to **0.2.0**; `CHANGELOG.md` added (Added /
Changed / Fixed for 0.2.0, one-paragraph 0.1.0). `dist/` rebuilt clean, `twine check` PASSED,
wheel installed in a fresh venv: `tabaudit 0.2.0`, no cleanlab, `gate` exits 1 on the demo
leak. Phase 6 renumbered to v0.3.0; 5.4 skipped. Release steps still to run after the go-ahead:
`git tag v0.2.0 && git push --tags`, `twine upload dist/*`, `gh release create v0.2.0`.

**2026-09-15 (released)** — **tabaudit 0.2.0 is on PyPI** (https://pypi.org/project/tabaudit/0.2.0/),
tag `v0.2.0` pushed, GitHub release created with CHANGELOG notes + wheel/sdist. Verified:
`pip install tabaudit==0.2.0` in a fresh venv → `tabaudit 0.2.0`, no cleanlab, `gate` works.
The README's `@v0.2.0` Action / pre-commit snippets now resolve. **Left: 2.5 (make repo
public) and 3.5 (LinkedIn post).** Then Phase 6 (`fix`, v0.3.0) or Project 2.

**2026-09-16 (comparison refresh)** — README "How it compares" updated: `ydata-profiling` was
renamed **`fg-data-profiling`** in Apr 2026 (now 4.20; the old package gets no more updates), so
the row is relabelled and re-linked. Added a fifth column for
**[`dataleaks`](https://pypi.org/project/dataleaks/) 0.1.2** — a leakage-only CLI first released
2026-09-06, ten days before tabaudit 0.1.0, i.e. the nearest competitor. Measured in a throwaway
venv (since deleted): adds **1 MB** (pandas only), **1.5 s** on `adult` vs tabaudit's 9.8 s,
works on pandas 3 / numpy 2.5. Probed with `benchmarks/inject.py` on `adult`: **0 findings on all
four injected faults** (duplicates, 3 % label flips, noisy target copy, missingness leak), and 0
on Titanic — its `target_statistical` detector is Pearson `|r| >= 0.999` and `feature_suspicious`
is a column-name regex + `|r| >= 0.90`, so nothing model-based. It *does* catch an exact target
copy (CRITICAL) and train/test row overlap (HIGH), but exits 0 regardless, so it can't gate CI.
Where it genuinely leads: **temporal, preprocessing (fit-before-split) and cross-dataset
leakage** — which is Phase 7 here, now with a competitor's taxonomy to aim at.

**2026-09-16 (6.9 + 6.10)** — Reviewed `../update.md` (an outside suggestion list) against the
repo and took the two items that survive contact with the plan: **6.9 `impact`** and **6.10
score breakdown**, both shipped and pushed. The rest is recorded above: idea 2 was already
Phase 6.5, its "rigorous evaluation" advice was already `docs/evaluation.md`, and ideas 4–6 are
declined with reasons. The framework reframe in that document is a thesis, not a 1–2-week
portfolio project, and adopting it would cost Projects 2 and 3. 57 tests pass, ruff clean.
`impact` also gave the README a row no competitor can tick. **Left: 2.5 (repo public), 3.5
(LinkedIn post), 6.11 (rebuild the GIF before 0.3.0), then the rest of Phase 6 (`fix`).**

**2026-09-16 (6.1–6.4: `tabaudit fix` works)** — `Fix(action, params, safe, flag)` on findings
(validated in `__post_init__`; unsafe fixes must name their flag), fixes emitted by
`duplicates` / `schema` / `leakage` / `label_noise`, `src/tabaudit/fix.py` with
`apply_fixes(df, report, flags) -> (clean_df, FixPlan)`, and the `tabaudit fix` command
(writes `<name>.clean.csv` + `<name>.fixplan.json`, input untouched, exit 0 with fixes
outstanding). Apply order: drop_columns → coerce_dtype → drop_rows → flag_rows, so the flag
column can never be dropped or coerced. Guards: the target is never dropped, a drop that would
leave no features is skipped, applying a plan twice is a no-op. 12 new tests (69 total).

**Two decisions worth remembering.** (1) ≥50 %-missing and near-constant columns emit **no fix
at all** rather than an extra flag — drop-or-impute is a judgement call, and keeping the flag
surface at exactly the two the design table names (`--drop-leaky`, `--flag-noise`) is worth
more than covering every finding. (2) Rows with no label *are* a safe fix (you cannot train on
an unlabeled row), even though it drops rows.

Numbers: demo 4 992 → 4 728 rows, 11 → 10 cols on the safe fixes; auditing train alone the
score goes **36 F → 46 D** safe-only and **36 F → 87 B** with both flags. On bank-marketing the
output is *unchanged: 45 211 rows, 17 columns* — both findings there are judgement calls, which
is the whole design in one transcript and now the worked example in `docs/fix.md`.

**Left in Phase 6:** 6.5 (`pipeline.py` — generate the `ColumnTransformer` skeleton), 6.6
(re-run benchmarks with `fix`, add a rows/cols-removed column), 6.7 (0.3.0 release: version
bump, build, upload, tag; `docs/fix.md` and the README section are already written), 6.8
(second LinkedIn post), plus 6.11 (rebuild `docs/demo.gif`).

**2026-09-16 (6.5: generated pipeline code)** — `src/tabaudit/pipeline.py` turns the cleaned
frame into `<name>_pipeline.py`: dtype-driven `ColumnTransformer` (scale / one-hot ≤ 20 levels
/ ordinal above / impute only where nulls exist), datetimes named but left out with a note,
`remainder="drop"`, and an `EXCLUDED` list carrying the flagged-but-kept columns *and the
finding that named them* — including `tabaudit_suspect`, which must never become a feature.
Written by `tabaudit fix` unless `--no-pipeline`.

It is a real file, not a sketch: the test suite executes it (`held-out accuracy: 0.816`,
`5-fold: 0.817 +/- 0.014` on the demo; R2 0.692 on a synthetic regression set). It also has to
survive the *user's* linter, so the generator wraps to 88 columns (not this repo's 100), emits
double-quoted literals and sorted imports, and a test runs `ruff check --isolated` and
`ruff format --check` over the output — which is what caught the one line that was too long.
9 new tests (78 total).

**Phase 6 is now functionally complete.** Left: 6.6 (re-run benchmarks with `fix`), 6.7 (0.3.0
release — `docs/fix.md` and the README section are written, so it is version bump + build +
upload + tag), 6.8 (LinkedIn post), 6.11 (rebuild `docs/demo.gif`).

**2026-09-16 (6.6: benchmarks re-run with `fix`)** — `run_benchmarks.py` now audits, applies the
safe fixes with no flags, re-audits, and records rows/cols removed plus the score after; it
prints a REGRESSION block if any safe fix ever lowers a score. All ten datasets, 2 min 50 s.

**Two results worth keeping.** (1) Every one of the ten *before* scores is byte-identical to the
published table, which is the real check that `impact` (INFO-only) cannot move a score. (2) Safe
fixes never lowered a score: five datasets improved — breast-w **82 B → 97 A**, spambase **75 B →
90 A**, creditcard 78 → 85, telco 80 → 86, adult 84 → 87 — and five were left exactly as they
were, including both datasets with the famous leaks. Every automatic gain is duplicates or one
dtype coercion; no column was dropped anywhere, because none of these curated sets ships a
constant or `Unnamed: 0` column. New section in `docs/benchmarks.md`, plus a README pointer.

**Left before 0.3.0:** 6.7 (version bump, build, upload, tag, release notes), 6.11 (rebuild
`docs/demo.gif`), 6.8 (LinkedIn post).

**2026-09-16 (6.7: 0.3.0 released)** — **tabaudit 0.3.0 is on PyPI**
(https://pypi.org/project/tabaudit/0.3.0/), tag `v0.3.0` pushed, GitHub release created with
notes + wheel/sdist. Verified *before* uploading, in a fresh venv off the built wheel: version
0.3.0, no cleanlab, `demo` shows the impact finding and the per-check bars, `fix` writes the
cleaned CSV + fix plan + pipeline, the generated pipeline runs (held-out accuracy 0.816), and
`gate --fail-on high` still exits 1 on the planted leak. Verified again after uploading with
`pip install tabaudit==0.3.0` from PyPI. README / `action.yml` / pre-commit pins all moved to
`v0.3.0`, so those snippets resolve against the new tag.

**Phase 6 is done.** Left on the whole project: **2.5** (make the repo public — your click),
**6.11** (rebuild `docs/demo.gif`; the verdict panel changed twice), **6.8 / 3.5** (the
LinkedIn posts). Note that until the repo is public, the README images on the PyPI page
(`raw.githubusercontent.com/.../docs/demo.gif`) cannot load for anonymous visitors — that has
been true since 0.1.0 and going public fixes it. Then: Project 2 (LLM → tiny model
distillation).

**2026-09-17 (6.11: demo GIF rebuilt)** — `docs/demo.gif` now shows 0.3.0: the per-check bars,
the INFO badge and the `impact` finding count. Three things had to change to get there, all in
the tape/script rather than the tool: (1) `tabaudit demo` takes ~35 s now (the `impact` check
fits two more models) and the tape only waited 14 s, so the first render was a blank prompt —
the tape now waits 45 s with the recorder hidden; (2) the recording shell inherited `NO_COLOR=1`
from the agent harness and Rich rendered everything monochrome — the script clears it;
(3) the script deleted and recreated its temp folder, which fails when any shell is sitting in
it — it now clears the contents instead. The 46-line cut and the "N more findings" trailer were
also updated for the 9-finding output. Left on the project: **2.5** (public repo), **3.5 / 6.8**
(LinkedIn posts). Then Project 2.
