# tabaudit — Roadmap

_Last updated: 2026-09-14. Companion to `../AI_ML_Portfolio_Projects.md` (the 3-project plan)._

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

## Phase 4 — Stretch (only if Phases 1–3 are finished)

- [ ] Group / time leakage check: entity IDs that appear in both train and test; date columns where test dates precede train dates
- [ ] Near-duplicate detection (numeric tolerance / fuzzy text)
- [ ] Label-noise for regression targets (residual-based)
- [ ] `pre-commit` hook / reusable GitHub Action

## Then → Project 2 (LLM → tiny model distillation)

Do **not** start until Phase 3 is complete. See `../AI_ML_Portfolio_Projects.md`.

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
