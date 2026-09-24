"""The version is written in two places; a release must bump both."""

from __future__ import annotations

import re
from pathlib import Path

import tabaudit


def test_version_matches_pyproject():
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M).group(1)
    assert tabaudit.__version__ == declared
