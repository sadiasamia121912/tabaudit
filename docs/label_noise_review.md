# Label-noise review (roadmap task 1.5)

The "row #" is only the line number in the file - it says nothing about the patient.
Judge each suspect on its *feature values* against the "typical" columns printed by
`python benchmarks/show_suspects.py <dataset> --top 5`.

Verdict is exactly one of:
- **looks mislabeled** - the features clearly match the *suggested* class, not the given one
- **plausible** - the features fit the *given* label; the model is probably wrong here
- **can't tell** - mixed signals

In `why`, name the 2-3 features that decided it (e.g. "no angina, oldpeak 0, thal 3").


## heart-statlog

| row # (file line) | given label | model suggests | verdict | why (which features?) |
|---|---|---|---|---|
| 69 | present | absent |  |  |
| 248 | present | absent |  |  |
| 258 | present | absent |  |  |
| 3 | absent | present |  |  |
| 187 | absent | present |  |  |

## diabetes

| row # (file line) | given label | model suggests | verdict | why (which features?) |
|---|---|---|---|---|
| 6 | tested_positive | tested_negative |  |  |
| 197 | tested_positive | tested_negative |  |  |
| 109 | tested_positive | tested_negative |  |  |
| 400 | tested_positive | tested_negative |  |  |
| 328 | tested_positive | tested_negative |  |  |
