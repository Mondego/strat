# label_runs.py — rationale and pre-registration

## Status

**Pre-registered.** Thresholds approved by the author on 2026-11-14 and committed in the
commit that adds this file, before `probes.py` and `dynamics.py` were run on the trained
checkpoints. Up to that commit only the hand-built self-test models had been probed; for
the trained checkpoints only the absence of NaNs was checked, and no probe values were
looked at. Any later change to a threshold must be dated below and reported as post hoc.

## Principle

Each failure mode in the paper makes a prediction about something measurable. A run
gets a label only if all of that mode's predictions hold. Labels are not exclusive. A
failed run that matches no rule is **Unclassified**. A large Unclassified share is a
result, not a bug: it means the taxonomy does not cover what the models do.

Units: the sign gap is 2 for a correct model and 0 when the operator has no effect; the
flip ratio is 1 when correct and 0 when ignored; gates are in [0, 1].

## Rules

| Label | Rule | What the paper claims it means |
|---|---|---|
| Success | OOD MAE < 100 | Same "Logic Acc" threshold as Section 6 |
| Linear Trap | \|sign gap\| < 0.5 on ID **and** OOD, **and** \|ID flip ratio\| < 0.25 | Fits values and ignores operators |
| Magnitude Leak | ID sign gap ≥ 1.0, \|OOD sign gap\| < 0.5, **and** removing magnitude from the gate (total ablation) cuts OOD MAE to ≤ 50% | Uses operators at training scale; magnitude reaching the gate breaks it OOD |
| Open Gate Trap | ID effective gate > 0.5 after both + and −, \|difference\| < 0.1, **and** clamping the gate to its mean raises ID MAE by ≤ 10% | Gates stay open and do no switching |
| Gradient Wall | attention is trainable, ID routing argmax < 0.5, median gate saturation over training > 0.9, **and** median attention-gradient norm < 0.1 × Config A's median | Saturated gates starve attention of gradient, so routing is never learned |
| Feature Collapse | OOD spread ratio std(ŷ)/std(target) < 0.25 while ID ratio ≥ 0.75 | Normalisation destroys magnitude, so OOD outputs stop scaling |
| Unclassified | failed (not Success) and none of the above | — |

## Where the numbers come from

- **0.5 / 1.0 on the sign gap**: a quarter and half of the ideal 2. Below 0.5 the
  operator changes the derivative by less than ±0.25 per operand. At ≥ 1.0 the model
  treats + and − clearly differently.
- **0.25 on the flip ratio**: flipping an operator moves the output by under a quarter of
  the correct amount.
- **50% for the leak**: the magnitude ablation must explain at least half the OOD error
  for magnitude to count as the cause, not just as correlated with it.
- **0.5 / 0.1 for open gates**: "open" means more than half conductive; "not switching"
  means the two operator contexts differ by under a tenth of the range. The clamp test
  adds the causal half: if fixing the gate does not hurt, it was not doing the work.
- **Gradient Wall**: routing below 0.5 means attention mostly fails to pick the operator.
  0.9 saturation follows the paper's "saturated gates" wording. The 10%-of-A rule makes
  "starved" relative to a model where the wall is not claimed; an absolute gradient
  threshold would depend on loss scale (As trains on targets divided by 5000).
- **Feature Collapse**: the paper's description (LayerNorm destroys absolute magnitude)
  predicts that OOD outputs stop growing with magnitude, so their spread shrinks relative
  to the targets'. 0.25 and 0.75 separate "scales" from "does not scale".
  The first draft used the slope of ŷ on the target. Testing the rules on the hand-built
  models (2026-11-14, before any trained model was probed) showed that the magnitude-leak
  model also satisfied it: any model that ignores signs at OOD has a near-zero slope on
  the signed target. The spread ratio does not have that problem: a sign-blind model
  still produces outputs of OOD size.

## Known gaps, decided in advance

- **Operating Point Overfit** has no rule. The paper describes it as low ID error with
  OOD failure after "fixing the signal-to-noise ratio", which is not a mechanism distinct
  from Magnitude Leak. Cb runs are expected to show up as Magnitude Leak or Unclassified.
  If so, the rebuttal should merge or withdraw the label.
- **Linear Trap vs. Open Gate overlap by construction**: a gate stuck fully open (or fully
  closed) also ignores operators. A run with both labels is consistent: the gate
  mechanism (Open Gate) explains the behaviour (Linear Trap).
- Thresholds are set by reasoning about units, not tuned on data. If a rule turns out to
  be badly placed, any change must be dated here and reported as post hoc, with both the
  original and changed results.
