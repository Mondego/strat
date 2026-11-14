# make_table2.py — rationale

## Why generate it

The submitted Table 2 could not be traced to any committed code (see `ablation.py.md`).
Generating the table from `results.csv` keeps the published numbers tied to a
reproducible run, and the table can be regenerated after any rerun.

## Choices

- **Text columns copied verbatim** from the submitted Table 2 (strategy, hypothesis,
  diagnosis), including "Open gate trap" in lower case and the row label "Bc" for the
  configuration Appendix B calls Cc. As of 2026-11-14 the diagnoses are deliberately
  unchanged: they are to be revisited once the per-seed mechanism probes exist.
- **Percentiles**: numpy's default (linear interpolation), over the 50 runs per config.
  ID MAE: 25th/50th/75th; OOD MAE: 50th/75th, as in the submitted table.
- **Hand-coded row hard-coded as zeros.** Checked 2026-11-14: `DenseBlock` with oracle
  attention and frozen logic gives MAE exactly 0.0 (max error 0.0) on both fixed
  evaluation sets.
- **STRAT-Lite not included**, matching the submitted table; its numbers are in
  `results.csv` and are quoted in Section 5.3 of the paper.
- **`\resizebox`** to fit the 9 columns in the ICLR text width; requires `graphicx` (loaded
  by the ICLR template) and `booktabs`.
