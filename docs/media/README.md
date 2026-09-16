# docs/media

Images used in the README, the docs and the release write-ups — and the script that builds
them.

| file | what it is |
|---|---|
| `v03_1_bank_unchanged.png` | `tabaudit fix` on bank-marketing, changing nothing on purpose |
| `v03_2_demo_applied.png` | `tabaudit fix` on the demo data: three fixes applied, three held back |
| `v03_3_pipeline_code.png` | the generated pipeline file, and why it is code and not a cleaned CSV |
| `tabaudit_demo_*.png` | `tabaudit demo` terminal report (0.2.0) |

Regenerate the `v03_*` set from the repo root:

```bash
python docs/media/make_post_images.py
```

It is self-contained — it writes the demo dataset and a renamed copy of bank-marketing into a
temp directory and runs the real audit / fix / pipeline code over them. Nothing is a mock-up:
`render_terminal.py` draws rich's own styled segments, so the colours and characters are the
ones the terminal prints. After drawing each transcript the script runs the actual CLI over the
same file and fails if a phrase shown in the image is missing from the CLI's output, so the
images cannot go stale without someone noticing.

Needs the dev extras (Pillow) and the OpenML cache under `benchmarks/openml_cache/`.
