# tabaudit — Roadmap

_Last updated: 2026-09-13. Companion to `../AI_ML_Portfolio_Projects.md` (the 3-project plan)._

## Where things stand

| Item | Status |
|------|--------|
| Core tool (5 checks, CLI, HTML/JSON report, `demo`) | ✅ v0.1.0 done |
| Tests (`pytest`) | ✅ 22 passing |
| Lint (`ruff`) | ✅ clean |
| GitHub Actions CI (Linux/Windows × Py 3.10/3.12/3.13) | ✅ configured |
| Pushed to GitHub (`sadiasamia121912/tabaudit`, **private**) | ✅ 1 commit |
| Benchmarks on real datasets | ⏳ **started** — 5 OpenML datasets are cached in `benchmarks/openml_cache/` but no script or results exist yet |
| README benchmark table + GIF | ❌ |
| Public repo | ❌ |
| PyPI release (`pip install tabaudit`) | ❌ |

**How to get running again (every session):**
```powershell
cd C:\Users\User\dev\tabaudit
.\.venv\Scripts\activate
pytest -q                      # should say "22 passed"
tabaudit demo                  # see every check fire on synthetic data
```

---

## Phase 1 — Real-world evidence  (2 days)  ← YOU ARE HERE

Goal: a table of *real* findings on datasets everyone recognises. This is the number
that goes on the résumé.

- [ ] **1.1** Write `benchmarks/run_benchmarks.py`
  - loads each dataset via `sklearn.datasets.fetch_openml(name, data_home="benchmarks/openml_cache")`
  - calls `tabaudit.run_audit(df, target=...)` on each
  - writes one row per dataset to `benchmarks/results.json` (score, grade, counts per severity, key findings)
  - Already cached (from last session): `titanic`, `adult`, `credit-g`, `creditcard`, `telco-customer-churn`
- [ ] **1.2** Add 3 more: `heart-disease` / `breast-w` (Breast Cancer Wisconsin), `bank-marketing`, `house_prices` (regression — tests the R² path)
- [ ] **1.3** Run it. Expect a few minutes for `creditcard` (285k rows → sampled to 50k).
- [ ] **1.4** Write `docs/benchmarks.md`: one table (dataset · rows · score · grade · headline finding) + a paragraph per interesting result
- [ ] **1.5** Manually verify 2–3 flagged label-noise rows (open the rows, look at the features, decide if the label really looks wrong). Screenshot → `docs/`. This is what makes the claim credible in an interview.
- [ ] **1.6** Commit: `git add benchmarks/run_benchmarks.py benchmarks/results.json docs/ && git commit`
  - do **not** commit `benchmarks/openml_cache/` — add it to `.gitignore`

**Done when:** `docs/benchmarks.md` has ≥ 8 datasets and you can say "found X in N of 8".

## Phase 2 — Polish  (1 day)

- [ ] **2.1** README: paste the benchmark table under a new "Results on real datasets" section
- [ ] **2.2** Record a ~20 s terminal GIF of `tabaudit demo` (use [vhs](https://github.com/charmbracelet/vhs) or asciinema+agg) → `docs/demo.gif`, embed in README
- [ ] **2.3** `tabaudit checks --explain` (or a `docs/checks.md` page): one paragraph per check on *how* it works and *why* the threshold is what it is
- [ ] **2.4** Tick the "Audit results for popular public benchmark datasets" box in the README roadmap
- [ ] **2.5** Make the repo **public** (GitHub → Settings → Danger zone → Change visibility)

## Phase 3 — Publish  (½ day)

- [ ] **3.1** Create a PyPI account + API token (https://pypi.org)
- [ ] **3.2** Dry run on TestPyPI first:
  ```powershell
  python -m build
  twine upload --repository testpypi dist/*
  pip install -i https://test.pypi.org/simple/ tabaudit   # in a fresh venv
  ```
- [ ] **3.3** Real upload: `twine upload dist/*`, then verify `pip install tabaudit && tabaudit --version` in a **fresh** venv
- [ ] **3.4** `git tag v0.1.0 && git push --tags`; write GitHub release notes (copy the README "What it checks" table)
- [ ] **3.5** LinkedIn/blog post: *"I audited 8 popular ML datasets — here's what's wrong with them"* (the benchmark table + 3 concrete examples)
- [ ] **3.6** Fill in the numbers in the résumé bullet in `../AI_ML_Portfolio_Projects.md`

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
