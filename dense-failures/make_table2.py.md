# make_table2.py — rationale

## Why generate it

The submitted Table 2 could not be traced to any committed code (see `ablation.py.md`).
Generating the table from the result CSVs ties the published numbers to a reproducible
run, and the table can be regenerated after any rerun.

## Layout (revised 2026-11-14)

The first version kept the submitted "Diagnosis / Mechanism" column verbatim. After the
mechanism probes (`label_runs.py.md`, `attention_test.py.md`) most of those diagnoses did
not survive, so the column was replaced by measured columns (medians over seeds):

- **ID slope**: slope of the in-distribution operator-flip effect regressed on the flipped
  operand (1 = proportional, as in the correct rule; ≈ 0 = fixed offset).
- **OOD flip**: median flip ratio at OOD (1 = operator used correctly; 0 = ignored).
- **Gate abl.**: OOD MAE when the gate receives its mean-operand input.

- **ID flip** (added 2026-11-14): median ID flip ratio, from `probes.csv`. Section 4.2
  contrasts the ID and OOD flip ratios, and without this column the ID half of that
  contrast was not in the table.

These columns were chosen because together they separate the two mechanisms found
(magnitude-reading gate vs. fixed offset) and they are defined for every row, including
the hand-coded reference. The Hypothesis column was dropped for width;
it remains in Table 9. No STRAT-Lite row: Table 2 sits in Section 4, before STRAT is
introduced, and STRAT-Lite's numbers are reported in Section 5.3.

## Choices

- **Percentiles**: numpy's default (linear interpolation), over the 50 runs per config.
- **Hand-coded row hard-coded.** Checked 2026-11-14: oracle attention + frozen logic gives
  MAE exactly 0.0 on both evaluation sets; its gate never reads D0, so the gate ablation
  leaves it at 0; its flip slope and ratio are 1.00 (probes self-test).
- **"Bc" renamed C$_c$**, matching Appendix B.
- **A$_{curr}$ marked with a dagger.** It trains on operands up to 5000, so the 50.5
  reference used by the slope regression and the gate ablation is not typical for it, and
  the ID set (operands in ID_RANGE) covers only the low end of its training range. Its
  median ID flip ratio is 5.03.
- Requires `booktabs`, and `graphicx` for `\resizebox` (both loaded by the paper).
