"""Render the images used in the 0.3.0 write-ups, from real tool output.

    python docs/media/make_post_images.py

Self-contained: it writes the demo dataset and a renamed copy of bank-marketing into a temp
directory, runs the real audit / fix / pipeline code over them, and draws rich's own styled
segments to PNG (see render_terminal.py). Nothing here is a mock-up, and each image shows the
output of the command printed inside it.

The two transcript images rebuild the console output from `cli._plan_table` plus a few lines
of surrounding text. That is a small duplication of `cli.fix`, so after each render the script
runs the *actual* CLI over the same file and checks that the phrases in the image are in its
output — if the CLI's wording changes, this fails loudly instead of quietly shipping a stale
picture. Needs the dev extras (Pillow) and the OpenML cache used by benchmarks/.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from rich.syntax import Syntax
from rich.text import Text
from sklearn.datasets import fetch_openml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "benchmarks"))

from render_terminal import render  # noqa: E402

from run_benchmarks import BANK_MARKETING_COLS  # noqa: E402  (single source of truth)
from tabaudit import run_audit  # noqa: E402
from tabaudit.cli import _plan_table  # noqa: E402
from tabaudit.fix import apply_fixes  # noqa: E402

WIDTH = 92


def prompt(cmd: str) -> Text:
    return Text.assemble(("$ ", "bold green"), (cmd, "bold white"))


def shape_line(plan) -> Text:
    unchanged = plan.rows_after == plan.rows_before and plan.cols_after == plan.cols_before
    if unchanged and not plan.applied:
        return Text.assemble(
            (f"unchanged: {plan.rows_after:,} rows, {plan.cols_after} columns", "bold white"),
            (" — nothing here has exactly one right answer", "bold yellow"),
        )
    return Text(
        f"{plan.rows_before:,} → {plan.rows_after:,} rows, "
        f"{plan.cols_before} → {plan.cols_after} columns",
        style="bold white",
    )


def flags_line(plan) -> Text:
    return Text.assemble(
        (f"{len(plan.skipped)} fix(es) need your say-so: ", "yellow"),
        (" ".join(plan.flags_offered), "bold yellow"),
    )


def real_cli(args: list[str], cwd: Path) -> str:
    """What the installed command actually prints - the thing the image claims to show."""
    done = subprocess.run(
        [sys.executable, "-m", "tabaudit.cli", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if done.returncode != 0:
        raise SystemExit(f"CLI failed: {done.stderr}")
    return " ".join(done.stdout.split())


def transcript(data: Path, target: str, out: str, title: str, test: Path | None = None) -> None:
    df = pd.read_csv(data)
    test_df = pd.read_csv(test) if test else None
    report = run_audit(df, target=target, test=test_df)
    _, plan = apply_fixes(df, report, target=target)

    cmd = f"tabaudit fix {data.name} --target {target}"
    if test:
        cmd += f" --test {test.name}"
    body = [
        prompt(cmd),
        Text(""),
        Text.assemble(
            ("Health score before fixing  ", "bold white"),
            (
                f"{report.score}/100 {report.grade}",
                "bold red" if report.score < 60 else "bold yellow",
            ),
        ),
        _plan_table(plan),
        Text(""),
        shape_line(plan),
    ]
    if plan.flags_offered:
        body.append(flags_line(plan))

    # the image must not drift away from the command it prints
    args = [a for a in cmd.split()[1:]]
    printed = real_cli(args, data.parent)
    for phrase in (str(plan.rows_after), f"{report.score}/100", *plan.flags_offered):
        if phrase.replace(",", "") not in printed.replace(",", ""):
            raise SystemExit(f"'{phrase}' is in the image but not in the CLI's own output")

    print("wrote", render(body, HERE / out, width=WIDTH, title=title))


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="tabaudit-media-"))
    from tabaudit.demo import write_demo

    train, test = write_demo(tmp)

    bunch = fetch_openml(
        "bank-marketing",
        version=1,
        data_home=str(ROOT / "benchmarks" / "openml_cache"),
        as_frame=True,
        parser="auto",
    )
    # OpenML anonymises this one (V1..V16); restore the documented names so the image shows
    # `duration` - the column the whole story is about - and not `V12`.
    bank = bunch.frame.rename(columns={f"V{i}": c for i, c in enumerate(BANK_MARKETING_COLS, 1)})
    bank_path = tmp / "bank-marketing.csv"
    bank.to_csv(bank_path, index=False)

    transcript(bank_path, "Class", "v03_1_bank_unchanged.png", "tabaudit fix — bank-marketing")
    transcript(train, "churn", "v03_2_demo_applied.png", "tabaudit fix — demo data", test=test)

    # The pipeline file the CLI itself just wrote, next to the cleaned demo data.
    code = (tmp / "churn_train_pipeline.py").read_text(encoding="utf-8")

    def between(a: str, b: str, start: int = 0) -> str:
        i = code.index(a, start)
        return code[i : code.index(b, i)].rstrip()

    why = between("A scaler, an encoder", '"""')
    why = "\n".join(("# " + line) if line else "#" for line in why.splitlines())
    # Three excerpts of one real file. Elisions are marked, never spliced silently.
    snippet = (
        "\n\n# ...\n\n".join(
            [
                why,
                between("# Present in the file", "\n\n\nnumeric_steps"),
                between("# remainder=", "\n\n\ndef build_model"),
            ]
        )
        + "\n"
    )
    out = render(
        [
            prompt("cat churn_train_pipeline.py"),
            Text(""),
            Syntax(
                snippet, "python", theme="github-dark", background_color="#0d1117", word_wrap=False
            ),
        ],
        HERE / "v03_3_pipeline_code.png",
        width=WIDTH,
        title="generated by tabaudit fix",
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
