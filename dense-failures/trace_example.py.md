# trace_example.py — rationale

## Why

A reviewer found Section 3 too dense, with all the detail in Appendix A. The paper now
follows one expression, 3 + 5 − 7, through the hand-coded block (table `tab:alu_example`).
This script produces the numbers in that table, so they come from the same weights as
the hand-coded reference row of Table 2 and not from a hand calculation.

## Choices

- **The model** is `DenseBlock(oracle_attention=True, frozen_logic=True)`, i.e. the
  weights of Appendix A as set by `set_oracle_attention` and `set_frozen_logic`. Its
  remaining parameters (GLU biases other than D5's, readout) are overwritten or frozen by
  those functions, so the output does not depend on the random initialisation.
- **The expression 3 + 5 − 7** was chosen by the authors. It contains a first number with
  no operator, a number after +, and a number after −, and its result (1) is small enough
  to check by hand.

## Results (2026-11-15)

- Attention from 7: 0.881 to the minus, 0.119 to the plus. Both operators get the same
  query–key score (100); the distance penalty (−1 vs −3) gives softmax weights
  e²/(1+e²) = 0.881 and 0.119. So 7 holds IsSub = 0.88 and IsAdd = 0.12.
- Attention from 5: 1.00 to the plus (3 carries no IsOp flag, so it gets score ≈ 0).
- 3 has no token before it and attends only to itself; it holds no operator flag.
- Gate on D5: σ(100·IsNum + 100·IsSub − 150). For 7: σ(38.1) ≈ 1. For 3 and 5: σ(−50) ≈ 0.
  The gate also opens at the minus token itself (σ(14.4)), but it copies D0 = 0 there.
- After the GLU, D5 = 7 at the last token and 0 elsewhere. ΣD0 = 15, ΣD5 = 7,
  y = 15 − 2·7 = 1.0.
- Operator tokens are also changed by attention (e.g. the plus holds IsAdd = 1.73), but
  their D0 and D5 stay 0, so they contribute nothing to y. The paper table omits these
  values.
