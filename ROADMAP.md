# tabaudit — Roadmap

_Last updated: 2026-09-15. Companion to `../AI_ML_Portfolio_Projects.md` (the 3-project plan)._

## Where things stand

| Item | Status |
|------|--------|
| Core tool (5 checks, CLI, HTML/JSON report, `demo`) | ✅ v0.1.0 done |
| Tests (`pytest`) | ✅ 29 passing |
| Lint (`ruff`) | ✅ clean |
| GitHub Actions CI (Linux/Windows × Py 3.10/3.12/3.13) | ✅ configured |
| Pushed to GitHub (`sadiasamia121912/tabaudit`, **private**) | ✅ |
| Benchmarks on real datasets | ✅ 10 datasets in `docs/benchmarks.md`, gap-based leakage fix verified on them |
| README benchmark table + GIF | ✅ |
| Public repo | ❌ your click: Settings → Danger zone → Change visibility |
| PyPI release (`pip install tabaudit`) | ✅ 0.1.0 live, verified in a fresh venv |

**How to get running again (every session):**
```powershell
cd C:\Users\User\dev\tabaudit
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

## Phase 2 — Polish  (1 day)  ✅ except 2.5

- [x] **2.1** README: paste the benchmark table under a new "Results on real datasets" section _(2026-09-14)_
- [x] **2.2** Record a ~20 s terminal GIF of `tabaudit demo` → `docs/demo.gif`, embed in README _(2026-09-14; rebuild with `powershell docs/make_demo_gif.ps1`)_
- [x] **2.3** `docs/checks.md`: how each check works, every threshold and why, what each misses; plus how the score is computed _(2026-09-14)_
- [x] **2.4** Tick the "Audit results for popular public benchmark datasets" box in the README roadmap _(2026-09-14)_
- [ ] **2.5** Make the repo **public** — _your click, when you have read `docs/checks.md` and `docs/label_noise_review.md`_ (GitHub → Settings → Danger zone → Change visibility)

## Phase 3 — Publish  (½ day)  ← YOU ARE HERE (3.5 + 2.5 left)

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

## Phase 4 — v0.2.0: `tabaudit fix`  (1–2 days, only after 2.5 + 3.5 are done)

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

- [ ] **4.1** `Fix` dataclass in `findings.py`: `action` (`drop_rows` | `drop_columns` | `flag_rows` | `coerce_dtype`), `params: dict`, `safe: bool`, `flag: str | None` (the CLI flag that enables an unsafe fix). Add optional `fix: Fix | None = None` to `Finding`. Include it in `to_dict()`. Tests: serialisation round-trip.
- [ ] **4.2** Emit fixes from the checks that can — `duplicates.py` (drop_rows, safe), `schema.py` (drop_columns / coerce_dtype, safe), `leakage.py` (drop_columns, **unsafe**, flag `--drop-leaky`), `label_noise.py` (flag_rows, unsafe, flag `--flag-noise` → adds a `tabaudit_suspect` bool column). `imbalance.py` emits **no** fix — its recommendation text is the fix. Tests: each check's fix has the right rows/columns on `demo` data.
- [ ] **4.3** `fix.py`: `apply_fixes(df, report, enabled_flags) -> (clean_df, FixPlan)`. Apply order matters: drop columns first, then drop rows, then flag rows. `FixPlan` records what was applied, what was skipped and why, row/col counts before/after. Tests: applying the plan twice is a no-op; skipped unsafe fixes are listed.
- [ ] **4.4** `tabaudit fix data.csv --target y [--drop-leaky] [--flag-noise] [--out clean.csv]`. Prints the plan (✔ applied / ? needs a flag), writes `<name>.clean.csv` + `<name>.fixplan.json`. Exit code 0 even when unsafe fixes are skipped — skipping is the correct behaviour, not an error.
- [ ] **4.5** `pipeline.py`: generate `<name>_pipeline.py` — a scikit-learn `ColumnTransformer` skeleton from the cleaned frame's dtypes: `StandardScaler` for numeric, `OneHotEncoder(handle_unknown="ignore")` for categoricals with ≤ 20 levels, `OrdinalEncoder` above that, `SimpleImputer` where nulls were found. Header comment explaining *why this is code and not a transformed CSV* (fit on train only). This is generated **text**, not applied transformation — keep it that way.
- [ ] **4.6** Re-run `benchmarks/run_benchmarks.py` with `fix` on the 10 datasets → add a "rows/cols removed by safe fixes" column to `docs/benchmarks.md`. Sanity check: score after `fix` ≥ score before on every dataset.
- [ ] **4.7** `docs/fix.md`: the table above + one worked example (bank-marketing `duration`). README section "Fixing what it finds". Bump to 0.2.0, `python -m build`, `twine upload`, tag, release.
- [ ] **4.8** Second LinkedIn post: *"v0.2: tabaudit now fixes what it finds — and why it refuses to fix some things"*.

**Done when:** `tabaudit fix` on `tabaudit demo` data drops the duplicates and constant column, leaves the leaky column in place with a clear message, and the generated pipeline file runs end-to-end on the clean CSV.

## Phase 5 — Stretch (only if Phase 4 is finished)

- [ ] Group / time leakage check: entity IDs that appear in both train and test; date columns where test dates precede train dates
- [ ] Near-duplicate detection (numeric tolerance / fuzzy text)
- [ ] Label-noise for regression targets (residual-based)
- [ ] `pre-commit` hook / reusable GitHub Action

## Then → Project 2 (LLM → tiny model distillation)

Do **not** start until Phase 3 is complete. Phase 4 is optional — decide after 3.5 whether v0.2 or Project 2 is the better use of the next week. See `../AI_ML_Portfolio_Projects.md`.

---

## Rules of thumb

- One commit per finished checkbox; push at the end of each session.
- Run `pytest -q && ruff check src tests examples` before every commit.
- If something is taking > 2× the estimate, cut scope, don't extend time.

---

## Session log

**2026-09-13** — Phase 1 tasks 1.1–1.4 done, pushed (`f7b68aa`). `.venv` and dataset cache
deleted to free space. **To resume:**
```powershell
cd C:\Users\User\dev\tabaudit
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
cd C:\Users\User\dev\tabaudit
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
