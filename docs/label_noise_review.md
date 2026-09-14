# Label-noise review (roadmap task 1.5)

_Reviewed 2026-09-14. The top 5 suspects on two datasets, judged by eye against the
"typical" row of each class as printed by `python benchmarks/show_suspects.py <dataset> --top 5`._

## Conclusion

**5 of the 10 top-ranked suspects look genuinely mislabeled; the other 5 are ambiguous;
none looked like a clear false alarm.** In every "looks mislabeled" case the row's key
clinical markers all sit on the *opposite* side from its label - e.g. heart-statlog row 69
is labelled "disease present" but has no exercise angina, ST depression 0, no blocked
vessels and a normal thallium scan. The ambiguous cases have a real risk factor pulling the
other way (BMI 37-45 on the diabetes rows; a high ST depression on heart row 248).

Caveat that keeps this honest: "looks unlike its class" is not proof the label is wrong.
The Pima diabetes label is *diagnosed within five years of the measurements*, so a young,
obese patient with normal glucose today can legitimately be a positive. What this review
shows is that the ranking is useful - every top suspect deserved a second look - not that
5 % of the labels are provably wrong.

## How to judge a row

The "row #" is only the line number in the file - it says nothing about the patient.
Judge each suspect on its *feature values* against the "typical" columns.

Verdict is exactly one of:
- **looks mislabeled** - the features clearly match the *suggested* class, not the given one
- **plausible** - the features fit the *given* label; the model is probably wrong here
- **can't tell** - mixed signals

For heart-statlog the decisive features are the last five: `exercise_induced_angina`
(1 = pain on exertion), `oldpeak` (ST depression; higher is worse), `slope` (2 is worse),
`number_of_major_vessels` (blocked vessels; more is worse), `thal` (3 normal, 7 defect).
For diabetes the decisive feature is `plas` (glucose; typical positive 140, typical
negative 107), then `mass` (BMI) and `age`. A 0 in `insu` or `skin` means *not measured*,
not zero.

## heart-statlog

| row # (file line) | given label | model suggests | verdict | why (which features?) |
|---|---|---|---|---|
| 69 | present | absent | looks mislabeled | all 5 markers healthy: no angina, oldpeak 0, slope 1, 0 vessels, thal 3; age 47, max HR 152 |
| 248 | present | absent | can't tell | oldpeak 3, slope 2 and resting ECG 2 point to disease; no angina, 0 vessels, thal 3 point away |
| 258 | present | absent | looks mislabeled | all 5 markers healthy (no angina, oldpeak 0, slope 1, 0 vessels, thal 3); only cholesterol 335 is high |
| 3 | absent | present | looks mislabeled | 4 of 5 markers say disease: angina, slope 2, 1 vessel, thal 7; max HR only 105 |
| 187 | absent | present | can't tell | 3 blocked vessels + thal 7 are the two strongest disease markers, but no angina, oldpeak 0.1, slope 1 |

## diabetes

| row # (file line) | given label | model suggests | verdict | why (which features?) |
|---|---|---|---|---|
| 6 | tested_positive | tested_negative | looks mislabeled | glucose 78 (below even typical negative 107), age 26, pedigree 0.25; nothing points to diabetes |
| 197 | tested_positive | tested_negative | looks mislabeled | glucose 107, BMI 22.9 (lean), age 23; only pedigree 0.68 is on the positive side |
| 109 | tested_positive | tested_negative | can't tell | glucose 95 and age 24 say negative, but BMI 37.4 is a real risk factor |
| 400 | tested_positive | tested_negative | can't tell | glucose 95 says negative, but skin and insulin are unmeasured (0) - too little evidence |
| 328 | tested_positive | tested_negative | can't tell | glucose 102 and age 23 say negative, but BMI 45.5 is very high |

**Tally:** 5 looks mislabeled, 5 can't tell, 0 plausible.
