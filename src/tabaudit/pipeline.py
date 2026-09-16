"""Generate leak-free preprocessing code, because a preprocessed *file* cannot be leak-free.

A scaler, an encoder and an imputer all learn something from the data they are fitted on - a
mean, a category set, a median. Fitted on a whole file, those statistics carry information
from the test rows into the numbers the model trains on, which is exactly the contamination
`tabaudit audit` reports. So tabaudit never writes transformed values anywhere; it writes a
`ColumnTransformer` inside a `Pipeline`, which scikit-learn fits on the training fold only.

This module produces *text*. It does not transform anything.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone

import pandas as pd

from tabaudit import __version__

# Above this many distinct values, one-hot encoding stops being reasonable (a 5 000-column
# matrix from one ID-ish column) and ordinal codes are the lesser evil for tree models.
LOW_CARD_MAX = 20


def column_groups(
    df: pd.DataFrame,
    target: str | None = None,
    exclude: Mapping[str, str] | None = None,
    low_card_max: int = LOW_CARD_MAX,
) -> dict[str, list[str]]:
    """Split the frame's columns into the groups that need different treatment."""
    skip = {target} | set(exclude or {})
    groups: dict[str, list[str]] = {"numeric": [], "low_card": [], "high_card": [], "datetime": []}
    for col in df.columns:
        if col in skip:
            continue
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            groups["datetime"].append(col)
        elif pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            groups["numeric"].append(col)
        elif s.nunique(dropna=True) <= low_card_max:
            groups["low_card"].append(col)
        else:
            groups["high_card"].append(col)
    return groups


def _q(value: str) -> str:
    """A double-quoted literal, because that is what the common formatters rewrite to."""
    return json.dumps(str(value))


# The generated file lands in someone else's repo, where the formatter's default width is 88
# rather than this project's 100. Wrap to the stricter one.
WIDTH = 88


def _clip(line: str) -> str:
    """Keep a generated comment inside the width; a finding title can be long."""
    return line if len(line) <= WIDTH else line[: WIDTH - 1] + "…"


def _assign(name: str, cols: list[str]) -> str:
    """`NAME = [...]`, wrapped one column per line as soon as the line would be too long."""
    inner = ", ".join(_q(c) for c in cols)
    if len(name) + len(inner) + 5 <= WIDTH:
        return f"{name} = [{inner}]"
    body = "\n".join(f"    {_q(c)}," for c in cols)
    return f"{name} = [\n{body}\n]"


def _steps_block(name: str, steps: list[tuple[str, str]]) -> str:
    """One `Pipeline([...])` assignment, pre-wrapped so no formatter wants to touch it."""
    lines = []
    for label, expr in steps:
        one = f"        ({_q(label)}, {expr}),"
        if len(one) <= WIDTH:
            lines.append(one)
        else:
            lines.append(f"        (\n            {_q(label)},\n            {expr},\n        ),")
    body = "\n".join(lines)
    return f"{name} = Pipeline(\n    [\n{body}\n    ]\n)"


def generate_pipeline(
    df: pd.DataFrame,
    target: str | None,
    task: str,
    *,
    data_name: str,
    exclude: Mapping[str, str] | None = None,
    low_card_max: int = LOW_CARD_MAX,
    version: str = __version__,
) -> str:
    """Return runnable scikit-learn code for ``df``, as a string.

    ``exclude`` maps a column to the reason it should not become a model input - a leak the
    user chose to keep in the file, or tabaudit's own ``tabaudit_suspect`` marker. Those
    columns stay in the data and are dropped in the *code*, where the decision is visible
    and one comment away from being reversed.
    """
    exclude = dict(exclude or {})
    g = column_groups(df, target, exclude, low_card_max)
    has_null = {name: bool(df[cols].isna().any().any()) for name, cols in g.items() if cols}
    classification = task == "classification"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    sklearn_imports = [
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.pipeline import Pipeline",
    ]
    if any(has_null.get(k) for k in ("numeric", "low_card", "high_card")):
        sklearn_imports.append("from sklearn.impute import SimpleImputer")
    encoders = []
    if g["numeric"]:
        encoders.append("StandardScaler")
    if g["low_card"]:
        encoders.append("OneHotEncoder")
    if g["high_card"]:
        encoders.append("OrdinalEncoder")
    if encoders:
        sklearn_imports.append(f"from sklearn.preprocessing import {', '.join(sorted(encoders))}")
    model = ""
    if target:
        model = (
            "HistGradientBoostingClassifier" if classification else "HistGradientBoostingRegressor"
        )
        sklearn_imports.append(f"from sklearn.ensemble import {model}")
        sklearn_imports.append(
            "from sklearn.model_selection import cross_val_score, train_test_split"
        )
    # Sorted so the generated file passes the user's own import linter.
    imports = ["import pandas as pd", *sorted(sklearn_imports)]

    out: list[str] = []
    out.append(f'''"""Preprocessing pipeline for {data_name} - generated by tabaudit {version}.

Written by `tabaudit fix` on {stamp}. A starting point to edit, not a black box.

WHY THIS IS CODE AND NOT A CLEANED, SCALED CSV
----------------------------------------------
A scaler, an encoder and an imputer each *learn* something from the data they
see: a mean and variance, a set of categories, a median. Fit them on the whole
file and those statistics carry information from your test rows into the numbers
the model trains on. The test score then flatters you - the same defect tabaudit
reports as train/test contamination, introduced by the act of cleaning.

That is why nothing here was applied to your data. `Pipeline` fits every step on
the training fold and only transforms the rest, including inside
cross-validation, so the held-out score below means what it says.
"""''')
    out.append("")
    out.append("from __future__ import annotations")
    out.append("")
    out.extend(imports)
    out.append("")
    out.append(f"DATA = {_q(data_name)}")
    if target:
        out.append(f"TARGET = {_q(target)}")
    out.append("")

    out.append(
        "# Scaled and median-imputed. Booleans are numeric here (0/1).\n"
        + _assign("NUMERIC", g["numeric"])
    )
    out.append(
        f"# One-hot encoded: at most {low_card_max} distinct values each.\n"
        + _assign("LOW_CARDINALITY", g["low_card"])
    )
    out.append(
        f"# Ordinal-coded: more than {low_card_max} distinct values, so one-hot would explode\n"
        "# the matrix. Fine for tree models; for a linear model, group the rare levels instead.\n"
        + _assign("HIGH_CARDINALITY", g["high_card"])
    )
    if g["datetime"]:
        out.append(
            "# Left out on purpose: a raw timestamp is rarely a useful input and often\n"
            "# encodes collection order. Derive year / month / dayofweek / deltas from these,\n"
            "# then add those to NUMERIC.\n" + _assign("DATETIME", g["datetime"])
        )
    if exclude:
        reasons = "\n".join(_clip(f"#   {c}: {why}") for c, why in sorted(exclude.items()))
        out.append(
            "# Present in the file but NOT used as model inputs, because tabaudit flagged them:\n"
            f"{reasons}\n"
            "# Delete an entry to train with it anyway - deliberately, not by accident.\n"
            + _assign("EXCLUDED", sorted(exclude))
        )
    out.append("")
    out.append("")

    # ---- the transformers ------------------------------------------------
    parts: list[str] = []
    MOST_FREQUENT = ("impute", 'SimpleImputer(strategy="most_frequent")')
    if g["numeric"]:
        steps = [("impute", 'SimpleImputer(strategy="median")')] if has_null.get("numeric") else []
        steps.append(("scale", "StandardScaler()"))
        out.append(_steps_block("numeric_steps", steps))
        parts.append('("num", numeric_steps, NUMERIC)')
    if g["low_card"]:
        steps = [MOST_FREQUENT] if has_null.get("low_card") else []
        steps.append(("encode", 'OneHotEncoder(handle_unknown="ignore", sparse_output=False)'))
        out.append(_steps_block("low_card_steps", steps))
        parts.append('("low", low_card_steps, LOW_CARDINALITY)')
    if g["high_card"]:
        steps = [MOST_FREQUENT] if has_null.get("high_card") else []
        steps.append(
            ("encode", 'OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)')
        )
        out.append(_steps_block("high_card_steps", steps))
        parts.append('("high", high_card_steps, HIGH_CARDINALITY)')
    out.append("")

    joined = ",\n        ".join(parts) if parts else ""
    out.append("# remainder='drop': anything not listed above never reaches the model by accident.")
    out.append("preprocess = ColumnTransformer(")
    out.append(f"    [\n        {joined},\n    ]," if parts else "    [],")
    out.append('    remainder="drop",')
    out.append(")")
    out.append("")
    out.append("")

    if target:
        out.append("def build_model():")
        out.append(
            '    """The whole recipe: preprocessing and estimator as one fittable object."""'
        )
        out.append("    return Pipeline(")
        out.append("        [")
        out.append('            ("preprocess", preprocess),')
        out.append(f'            ("model", {model}(random_state=0)),')
        out.append("        ]")
        out.append("    )")
        out.append("")
        out.append("")
        out.append('if __name__ == "__main__":')
        out.append("    df = pd.read_csv(DATA)")
        drop_expr = "[TARGET, *EXCLUDED]" if exclude else "[TARGET]"
        out.append(f"    X, y = df.drop(columns={drop_expr}), df[TARGET]")
        strat = ", stratify=y" if classification else ""
        out.append(
            "    X_train, X_test, y_train, y_test = train_test_split(\n"
            f"        X, y, test_size=0.2, random_state=0{strat}\n"
            "    )"
        )
        out.append("")
        out.append("    model = build_model().fit(X_train, y_train)")
        scorer = "accuracy" if classification else "R2"
        out.append(f'    print(f"held-out {scorer}: {{model.score(X_test, y_test):.3f}}")')
        out.append("    # The same object cross-validates without leaking: every fold refits the")
        out.append("    # preprocessing on that fold's training rows only.")
        out.append("    scores = cross_val_score(build_model(), X, y, cv=5)")
        out.append('    print(f"5-fold: {scores.mean():.3f} +/- {scores.std():.3f}")')
    else:
        out.append('if __name__ == "__main__":')
        out.append("    df = pd.read_csv(DATA)")
        out.append("    out = preprocess.fit_transform(df)")
        out.append('    print(f"{df.shape} -> {out.shape} after preprocessing")')
    out.append("")
    return "\n".join(out)
