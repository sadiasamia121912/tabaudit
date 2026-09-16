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


def test_fail_on_severity(tmp_path):
    # The demo data has a CRITICAL leak (churn_reason); the imbalance check alone finds nothing CRITICAL.
    train, _ = write_demo(tmp_path)
    base = ["audit", str(train), "-t", "churn", "-q"]
    assert runner.invoke(app, [*base, "-c", "leakage", "--fail-on", "high"]).exit_code == 1
    assert runner.invoke(app, [*base, "-c", "leakage", "--fail-on", "CRITICAL"]).exit_code == 1
    ok = runner.invoke(app, [*base, "-c", "imbalance", "--fail-on", "critical"])
    assert ok.exit_code == 0, ok.output
    # A high bar passes; the --fail-under bar can still fail independently.
    assert (
        runner.invoke(
            app, [*base, "-c", "imbalance", "--fail-on", "critical", "--fail-under", "99"]
        ).exit_code
        == 1
    )


def test_fail_on_rejects_unknown_severity(tmp_path):
    train, _ = write_demo(tmp_path)
    res = runner.invoke(app, ["audit", str(train), "-t", "churn", "--fail-on", "severe"])
    assert res.exit_code == 2


def test_gate_reports_each_file_and_fails_on_the_leak(tmp_path):
    train, test = write_demo(tmp_path)
    res = runner.invoke(
        app, ["gate", str(train), str(test), "-t", "churn", "-c", "schema,leakage"]
    )  # default --fail-on high; the demo plants a CRITICAL leak
    assert res.exit_code == 1
    assert res.output.count("✖") == 2
    assert "critical" in res.output


def test_gate_passes_clean_checks_and_can_gate_on_score_only(tmp_path):
    train, _ = write_demo(tmp_path)
    ok = runner.invoke(app, ["gate", str(train), "-c", "imbalance"])
    assert ok.exit_code == 0 and "✔" in ok.output
    # --fail-on none: only the score matters
    assert (
        runner.invoke(app, ["gate", str(train), "-c", "schema", "--fail-on", "none"]).exit_code == 0
    )
    assert (
        runner.invoke(
            app, ["gate", str(train), "-c", "schema", "--fail-on", "none", "--fail-under", "99"]
        ).exit_code
        == 1
    )


def test_gate_unreadable_file_is_exit_2(tmp_path):
    train, _ = write_demo(tmp_path)
    res = runner.invoke(app, ["gate", str(train), str(tmp_path / "missing.csv"), "-c", "schema"])
    assert res.exit_code == 2
    assert "missing.csv" in res.output


def test_missing_file_is_a_clean_error():
    res = runner.invoke(app, ["audit", "nope.csv"])
    assert res.exit_code == 2
    assert "No such file" in res.output


def test_checks_command_lists_all():
    res = runner.invoke(app, ["checks"])
    assert res.exit_code == 0
    for name in ("schema", "duplicates", "imbalance", "leakage", "impact", "label_noise"):
        assert name in res.output
