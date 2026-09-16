"""Check registry. Each check is a callable (AuditContext) -> list[Finding]."""

from __future__ import annotations

from collections.abc import Callable

from tabaudit.checks import duplicates, imbalance, impact, label_noise, leakage, schema
from tabaudit.context import AuditContext
from tabaudit.findings import Finding

CheckFn = Callable[[AuditContext], list[Finding]]

# Ordered: cheap structural checks first, model-based checks last.
REGISTRY: dict[str, CheckFn] = {
    "schema": schema.run,
    "duplicates": duplicates.run,
    "imbalance": imbalance.run,
    "leakage": leakage.run,
    # `impact` prices what `leakage` flagged, so it must see ctx.excluded_features already set.
    "impact": impact.run,
    "label_noise": label_noise.run,
}

__all__ = ["REGISTRY", "CheckFn"]
