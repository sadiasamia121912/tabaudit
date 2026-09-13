from __future__ import annotations

import json

from typer.testing import CliRunner

from tabaudit.cli import app
from tabaudit.demo import write_demo

runner = CliRunner()


def test_audit_writes_html_and_json(tmp_path):
    train, test = write_demo(tmp_path)
    html, js = tmp_path / "r.html", tmp_path / "r.json"
    res = runner.invoke(
        app,
        [
            "audit",
            str(train),
            "--target",
            "churn",
            "--test",
            str(test),
            "--html",
            str(html),
            "--json",
            str(js),
            "--checks",
            "schema,duplicates,leakage",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Data Health Score" in res.output
    assert html.exists() and "tabaudit report" in html.read_text(encoding="utf-8")
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["grade"] in "ABCDF"
    assert any(f["check"] == "leakage" for f in data["findings"])


def test_fail_under_returns_1(tmp_path):
    train, _ = write_demo(tmp_path)
    res = runner.invoke(
        app,
        ["audit", str(train), "-t", "churn", "-c", "schema,leakage", "--fail-under", "90", "-q"],
    )
    assert res.exit_code == 1


def test_missing_file_is_a_clean_error():
    res = runner.invoke(app, ["audit", "nope.csv"])
    assert res.exit_code == 2
    assert "No such file" in res.output


def test_checks_command_lists_all():
    res = runner.invoke(app, ["checks"])
    assert res.exit_code == 0
    for name in ("schema", "duplicates", "imbalance", "leakage", "label_noise"):
        assert name in res.output
