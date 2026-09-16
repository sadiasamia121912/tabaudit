"""Apply the fixes a report asked for - and refuse, loudly, to guess at the rest.

The rule this module exists to enforce: a dataset edit is applied automatically only when
there is exactly one defensible answer. Dropping a duplicated row, a constant column or a
row with no label is arithmetic. Dropping a suspected leak, or acting on a suspected label
error, is a judgement about the world - so those wait for a flag, and when the flag is
absent they are *reported as skipped*, never silently ignored.

What this module deliberately cannot do is scale, encode or impute. Those must be fit on
the training fold inside a pipeline; writing them into a CSV bakes test-set statistics into
training data, which is the exact failure `tabaudit audit` exists to catch. See
`docs/fix.md`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from tabaudit.findings import AuditReport, Finding

# Applied in this order: columns go first so row work is cheaper and dtype coercion sees the
# final set of columns; flagging is last so the flag column itself is never dropped or coerced.
ORDER = ("drop_columns", "coerce_dtype", "drop_rows", "flag_rows")


@dataclass
class FixStep:
    """One fix, and what actually happened to it."""

    check: str
    title: str
    action: str
    safe: bool
    applied: bool
    flag: str | None = None
    note: str = ""  # why it was skipped, or what it did
    n_rows: int = 0
    n_columns: int = 0
    columns: list[str] = field(default_factory=list)


@dataclass
class FixPlan:
    steps: list[FixStep]
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int

    @property
    def applied(self) -> list[FixStep]:
        return [s for s in self.steps if s.applied]

    @property
    def skipped(self) -> list[FixStep]:
        return [s for s in self.steps if not s.applied]

    @property
    def flags_offered(self) -> list[str]:
        """Flags that would change the outcome if the user passed them."""
        return sorted({s.flag for s in self.skipped if s.flag})

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "cols_before": self.cols_before,
            "cols_after": self.cols_after,
            "n_applied": len(self.applied),
            "n_skipped": len(self.skipped),
            "flags_offered": self.flags_offered,
            "steps": [asdict(s) for s in self.steps],
        }


def _fixable(report: AuditReport) -> list[Finding]:
    out = [f for f in report.sorted_findings() if f.fix is not None]
    return sorted(out, key=lambda f: ORDER.index(f.fix.action))  # type: ignore[union-attr]


def apply_fixes(
    df: pd.DataFrame,
    report: AuditReport,
    enabled_flags: Iterable[str] = (),
    target: str | None = None,
) -> tuple[pd.DataFrame, FixPlan]:
    """Return a cleaned copy of ``df`` plus the record of what was and was not done.

    ``enabled_flags`` are the CLI flags the user passed (``--drop-leaky``, ``--flag-noise``).
    An unsafe fix whose flag is absent is skipped and listed in the plan; that is the
    correct outcome, not an error.
    """
    flags = set(enabled_flags)
    out = df.copy()
    rows_before, cols_before = len(out), out.shape[1]
    target = target or report.summary.target
    steps: list[FixStep] = []

    for finding in _fixable(report):
        fix = finding.fix
        assert fix is not None  # _fixable guarantees it
        step = FixStep(
            check=finding.check,
            title=finding.title,
            action=fix.action,
            safe=fix.safe,
            applied=False,
            flag=fix.flag,
            columns=fix.columns,
        )

        if not fix.safe and fix.flag not in flags:
            step.note = f"needs {fix.flag}"
            steps.append(step)
            continue

        if fix.action == "drop_columns":
            cols = [c for c in fix.columns if c in out.columns and c != target]
            remaining = [c for c in out.columns if c not in cols and c != target]
            if not cols:
                step.applied, step.note = True, "already applied"
            elif not remaining:
                step.note = "would leave no feature columns"
            else:
                out = out.drop(columns=cols)
                step.applied, step.n_columns, step.columns = True, len(cols), cols
                step.note = f"dropped {len(cols)} column(s)"

        elif fix.action == "coerce_dtype":
            cols = [c for c in fix.columns if c in out.columns]
            for col in cols:
                s = out[col]
                if pd.api.types.is_numeric_dtype(s):
                    continue  # already coerced - keeps a second pass a no-op
                out[col] = pd.to_numeric(
                    s.astype(str).str.replace(",", "", regex=False), errors="coerce"
                )
            step.applied, step.n_columns, step.columns = True, len(cols), cols
            step.note = f"coerced {len(cols)} column(s) to numeric" if cols else "already applied"

        elif fix.action == "drop_rows":
            rows = [r for r in fix.rows if r in out.index]
            if rows:
                out = out.drop(index=rows)
                step.applied, step.n_rows = True, len(rows)
                step.note = f"dropped {len(rows):,} row(s)"
            else:
                step.applied, step.note = True, "already applied"

        elif fix.action == "flag_rows":
            col = str(fix.params.get("column", "tabaudit_suspect"))
            rows = [r for r in fix.rows if r in out.index]
            if col not in out.columns:
                out[col] = False
            out.loc[rows, col] = True
            step.applied, step.n_rows, step.columns = True, len(rows), [col]
            step.note = f"flagged {len(rows):,} row(s) in '{col}'"

        steps.append(step)

    return out, FixPlan(
        steps=steps,
        rows_before=rows_before,
        rows_after=len(out),
        cols_before=cols_before,
        cols_after=out.shape[1],
    )
