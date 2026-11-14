# probes.py — rationale

Reviewers R1/R2: the failure taxonomy was an interpretation of error numbers, not a
measurement. These probes measure, per trained model, the quantities each named failure
mode makes a prediction about. Labels are then assigned by `label_runs.py`.

## Inputs

The fixed ID (1–100) and OOD (2000–5000) evaluation sets from `ablation.py`, 1000
sequences each: the same data Table 2 is computed on. "Number tokens" are positions with
IsNum = 1; padding is never probed. A number's sign is −1 if the token before it is a
minus, else +1 (the first number is +1).

## Probes (each computed on ID and on OOD)

| Column | Definition | Correct model |
|---|---|---|
| `mae` | MAE of the prediction | ≈ 0 |
| `sens_plus`, `sens_minus` | mean ∂ŷ/∂v over numbers after + / after − (autograd w.r.t. the raw operand) | +1 / −1 |
| `sign_gap` | `sens_plus − sens_minus` | 2 |
| `gate_plus`, `gate_minus` | mean *effective gate* at numbers after + / − | depends on attractor; must differ |
| `gate_diff` | `gate_plus − gate_minus` | ±1 |
| `gate_saturated` | share of gate units at number tokens with σ < 0.05 or > 0.95 (unweighted) | high |
| `route_mass`, `route_argmax` | attention from each non-first number to the token right before it (its operator): mean weight, and share where it is the argmax | 1, 1 |
| `clamp_mae` | MAE with every gate unit at number tokens fixed to its mean over ID number tokens | ≫ `mae` |
| `d0_direct_mae` | MAE with gate-input dimension 0 at number tokens replaced by its ID mean (DenseBlock only; STRAT's gate never sees D0) | = `mae` |
| `d0_total_mae` | MAE when the gate receives the input it would get if every operand were 50.5, while the value branch sees the real operands | = `mae` |
| `flip_ratio` | swap one operator per sequence (chosen with seed 777), numbers fixed; median of Δŷ / (∓2v) | 1 |
| `r2_blind`, `r2_aware` | R² of ŷ against Σ\|v\| and against the true signed sum | low, 1 |
| `slope_on_target` | regression slope of ŷ on the true target | 1 |
| `spread_ratio` | std(ŷ) / std(target) | 1 |

**Effective gate.** The GLU has one gate unit per written dimension (6 for DenseBlock;
D0 and D5 for STRAT-Lite). Unit *j* matters for the output in proportion to
|W_out[j]| · mean|v_j| over ID-set number tokens (v = value branch). The effective gate
is the weighted mean of the unit activations with those weights. Averaging the units
unweighted would let units with no path to the output dominate.

**Direct vs. total magnitude ablation.** In DenseBlock, attention's V projection mixes
D0 into every dimension before the GLU, so the gate can read magnitude through
dimensions other than 0. The direct ablation underestimates that path; the total one
removes every route from operand values into the gate. Both are kept because the gap
between them is itself informative (rung 1 vs. rung 2 of the planned constraint ladder).

## Validation (`--selftest`)

Four hand-built models with known mechanisms, built from the Appendix A ALU: correct;
gate forced open (bias +1000); gate forced closed (bias −1000); gate reading magnitude
(weight −0.05 from D0, so large operands close it). All 25 expected probe values were
recovered (2026-11-14), for example: correct → sign gap 2.00 ID and OOD, flip ratio 1.00,
routing 1.00; magnitude leak → sign gap 2.00 ID and 0.00 OOD, and both D0 ablations
bring OOD MAE to 0.

## Known limitations

- Sensitivities are local gradients; for a saturated sigmoid gate the derivative through
  the gate is ≈ 0, so `sens_*` measures the value path at the current operating point.
  `flip_ratio` is a finite perturbation and complements it.
- The clamp and the direct D0 ablation act only at number tokens. Operator tokens also
  pass through the GLU, and in DenseBlock their gate input can carry magnitude via
  attention; only the total ablation covers that.
