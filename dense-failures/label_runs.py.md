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

## Results and problems found (2026-11-14, first run on the 650 trained runs)

Thresholds unchanged. Output: `results/ablation/labels.csv`, `label_summary.csv`.

- **Linear Trap: 0% of runs in every config.** In-distribution, dense models do use the
  operators: median ID flip ratio ≈ 1.0 for every dense config except Bfl (0.81) and
  Acurr. The rule as written cannot fire. The paper's claim that A "ignores operators"
  is contradicted by the measurement.
- **Magnitude Leak**: 100% of Bfa and 90% of Cb runs (oracle attention: removing
  magnitude from the gate cuts OOD MAE from ~11,800 to ~1,000 and ~160). 20–26% of A,
  Bortho, Cg, Cbi. Cb's "Operating Point Overfit" is measured as Magnitude Leak, as
  anticipated above.
- **Gradient Wall: 0% of dense runs.** Bfl's median attention gradient is 0.72× Config A's,
  not starved, and its routing argmax (0.75) matches A's (0.76).
- **Open Gate Trap**: ≤ 16% anywhere. Median effective gates are 0.3–0.46 and equal
  after + and −, i.e. the gates are half-open and do not switch, but the "> 0.5" rule
  is not met.
- **Unclassified dominates** (34–90% of dense runs).

Post-hoc observation, not a pre-registered rule: 79% of dense runs have an ID flip
ratio in [0.75, 1.25] and an OOD |flip ratio| < 0.25, i.e. they use the operator
correctly at training scale and ignore it at OOD scale. Since their gates do not differ
between + and −, the sign must be carried by another path. Attention is the obvious
candidate: Q/K see D0, so attention weights depend on magnitude. Not yet tested.

Flaws in the pre-registered measures, reported rather than fixed:

1. **Gradient-norm comparison is scale-confounded.** Config A's attention gradients are
   large because D0 (values up to 100) enters Q/K. STRAT-Lite's attention never sees D0,
   so its gradients are ~1000× smaller and 4 of its 8 failed seeds (1019, 1020, 1025,
   1036; routing argmax 0) were labelled Gradient Wall partly for that reason.
2. **Saturation is unweighted across gate units.** In Bfl, 5 of the 6 frozen gate units
   have zero weights and sit at σ(0) = 0.5, so median saturation reads 0.17 even though
   the one unit that matters is saturated. The effective-gate weighting should be used.
3. **Routing argmax has no baseline.** With the ALiBi penalty and the causal mask, the
   immediately preceding token is favoured even without learning. A, Acurr, Bfl, Bortho,
   Cg and Cbi sit at a median of 0.75–0.76 whatever their outcome; an untrained-model
   baseline is needed. (Corrected 2026-11-18: an earlier version said all dense configs
   sit at ~0.75. Aln is 0.65, Bfa and Cb are fixed at 1.00, and As, Csb and Cc are 0.00.)
4. **Csb and Cc nearly meet Gradient Wall** (checked 2026-11-18). In 80% of Csb runs and
   70% of Cc runs, routing argmax < 0.5 and median gate saturation > 0.9; only the
   gradient condition fails, with median attention gradient 37× and 23× Config A's. The
   attention gain multiplies that gradient (100 in Csb, init 50 in Cc); divided by it the
   ratio is still ≈ 0.4–0.5, above the 0.1 threshold.
