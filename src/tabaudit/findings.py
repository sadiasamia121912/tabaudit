"""Data model shared by all checks and reporters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def rank(self) -> int:
        return {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}[self.value]

    @property
    def penalty(self) -> int:
        """Points deducted from the 100-point health score."""
        return {"critical": 30, "high": 15, "medium": 7, "low": 3, "info": 0}[self.value]


ACTIONS = frozenset({"drop_rows", "drop_columns", "flag_rows", "coerce_dtype"})


@dataclass
class Fix:
    """The one concrete edit a finding knows how to make to the dataset.

    ``safe`` means there is exactly one defensible answer - dropping a constant column,
    coercing a numeric column stored as text, removing a row with no label. Everything else
    carries ``safe=False`` and a ``flag``: the user has to say yes, because the right answer
    depends on knowledge the tool does not have (when a column becomes known, whether a
    suspect row is really mislabeled). A finding with no fix is one where even asking would
    be misleading - class imbalance is a modelling decision, not a dataset edit.
    """

    action: str  # one of ACTIONS
    params: dict[str, Any] = field(default_factory=dict)
    safe: bool = True
    flag: str | None = None  # the CLI flag that enables an unsafe fix

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"Unknown fix action '{self.action}'. Known: {sorted(ACTIONS)}")
        if not self.safe and not self.flag:
            raise ValueError("An unsafe fix must name the flag that enables it.")

    @property
    def rows(self) -> list:
        return list(self.params.get("rows", []))

    @property
    def columns(self) -> list[str]:
        return list(self.params.get("columns", []))


@dataclass
class Finding:
    check: str
    severity: Severity
    title: str
    detail: str
    recommendation: str = ""
    columns: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    fix: Fix | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass
class DatasetSummary:
    path: str
    n_rows: int
    n_cols: int
    target: str | None
    task: str  # "classification" | "regression" | "unsupervised"
    n_classes: int | None
    test_path: str | None = None
    n_test_rows: int | None = None
    memory_mb: float = 0.0
    dtypes: dict[str, int] = field(default_factory=dict)  # dtype kind -> count


@dataclass
class CheckRun:
    name: str
    status: str  # "ok" | "skipped" | "error"
    duration_s: float
    n_findings: int
    note: str = ""


@dataclass
class AuditReport:
    summary: DatasetSummary
    findings: list[Finding]
    checks: list[CheckRun]
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    )
    version: str = "0.1.0"

    # -- scoring -------------------------------------------------------------
    @property
    def score(self) -> int:
        penalty = sum(f.severity.penalty for f in self.findings)
        return max(0, 100 - penalty)

    @property
    def score_breakdown(self) -> dict[str, int]:
        """The same 0-100 scale, per check: what each one cost, in registry order.

        A check that ran and found nothing scores 100. The overall score is *not* the mean
        of these - it is 100 minus every penalty, so one bad check can sink it on its own.
        Checks that errored or were not selected are absent.
        """
        out: dict[str, int] = {}
        for run in self.checks:
            if run.status != "ok":
                continue
            penalty = sum(f.severity.penalty for f in self.findings if f.check == run.name)
            out[run.name] = max(0, 100 - penalty)
        return out

    @property
    def grade(self) -> str:
        s = self.score
        if s >= 90:
            return "A"
        if s >= 75:
            return "B"
        if s >= 60:
            return "C"
        if s >= 40:
            return "D"
        return "F"

    @property
    def verdict(self) -> str:
        return {
            "A": "Ready to train",
            "B": "Minor issues — review before training",
            "C": "Fix issues before trusting any metric",
            "D": "Results from this dataset will be misleading",
            "F": "Do not train on this dataset as-is",
        }[self.grade]

    def sorted_findings(self) -> list[Finding]:
        return sorted(self.findings, key=lambda f: (f.severity.rank, f.check, f.title))

    def count(self, sev: Severity) -> int:
        return sum(1 for f in self.findings if f.severity == sev)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tabaudit_version": self.version,
            "generated_at": self.generated_at,
            "score": self.score,
            "grade": self.grade,
            "verdict": self.verdict,
            "score_breakdown": self.score_breakdown,
            "summary": asdict(self.summary),
            "checks": [asdict(c) for c in self.checks],
            "findings": [f.to_dict() for f in self.sorted_findings()],
        }
