# type_norms.py — rationale

## Why

A reviewer asked whether the different norms of the learned Plus and Minus embeddings
in the paper's Table 3 (‖Plus‖ = 1.14, ‖Minus‖ = 1.62) have any confirmed effect. The
paper had no test of this. Two tests were run (2026-11-14):

1. **Descriptive, across seeds.** Is Minus longer in every run, or only in the run shown?
   Is the longer embedding the one for the operator that triggers the GLU?
2. **Intervention.** At evaluation, give both operators the same norm (direction kept)
   and measure the change in error and in the flip ratio.

## Choices

- **Training** follows the `__main__` of `strat_lite_learned_types.py` exactly: Adam 0.01,
  2001 steps of fresh batches of 64, clipping at 1.0, `torch.manual_seed(seed)` and
  `random.seed(seed)`. That is the setting of Section 5.5.1 and Table 3.
- **Seeds 1000–1009** match the seeds of the other experiments. **Seed 42** is included
  because it is the seed of `strat_lite_learned_types.py` and so should reproduce Table 3.
  It does: Plus = (0.50, −0.59, −0.67, −0.51), identical to Table 3, and
  Minus = (−0.81, 0.81, 0.83, 0.80), against (−0.80, 0.81, 0.83, 0.79) in Table 3.
- **Evaluation** uses the fixed ID/OOD sets of `dense-failures/ablation.py`, converted
  to token ids and values, and the operator flips of `probes.py` (seed 777), so that the
  numbers are comparable with Section 4 and Appendix B. Padding maps to Literal with
  value 0, the convention of the training batches.
- **Trigger operator.** This is the operator after which the GLU's contribution to the
  output, divided by the operand, is larger in magnitude on the ID set. In every run, one
  operator gives ≈ ±2 and the other ≈ 0, except seed 1002 (1.64 and −0.36).
- **Rescaling** changes one row of the embedding table, at evaluation only. Two
  directions are tested: the longer embedding shrunk to the shorter norm, and the shorter
  one grown to the longer norm.

## Results (2026-11-14, 11 runs)

- **Minus is longer than Plus in all 11 runs.** The ratio ranges from 1.22 to 2.66. This
  does not depend on which operator triggers the GLU: Plus triggers in 8 runs (negative
  readout on D0) and Minus in 3 (positive readout). The guess that the triggering operator
  gets the longer embedding is false.
- **Cosine(Plus, Minus)** ranges from −0.52 to −0.99, with median −0.92. Seed 42
  (−0.99) is the most antipodal run, which is the one shown in Table 3.
- **10 of 11 runs succeed** (OOD MAE < 100). Seed 1002 has OOD MAE 119. The median OOD
  MAE over seeds 1000–1009 is 44.4.
- **Growing Plus to Minus's norm changes nothing.** OOD MAE stays within a few units, and
  the flip ratio stays at 1.00 in all runs.
- **Shrinking Minus to Plus's norm breaks most runs.** It does so in three degrees:
  - **6 of 11 runs** (seeds 1002, 1003, 1004, 1006, 1007, 1008): OOD MAE rises to
    8,600–11,600 and the OOD flip ratio falls to 0.04–0.36.
  - **3 runs** (seeds 42, 1000, 1001): OOD MAE rises to 880–2,130, with flip ratio
    0.90–0.97.
  - **2 runs** (seeds 1005 and 1009, which have the smallest norm ratios, 1.32 and 1.22):
    OOD MAE stays below 100 (76 → 46 and 48 → 62).

**Reading.** The norm difference has a functional effect, and it is one-sided. The models
need Minus to be long: shrinking it removes the operator's effect. They do not need Plus
to be short: making it as long as Minus is harmless. Equal norms therefore work, provided
they are at Minus's length.

## Why Minus (added 2026-11-14)

**The first number behaves like a number after Plus.** In all 11 runs, the GLU's
contribution for the first number of a sequence, which has no operator, equals its
contribution after Plus: ≈ 2 in the 8 Plus-trigger runs, ≈ 0 in the 3 Minus-trigger runs,
and 1.64 for both in seed 1002. So every run has one gate state for "no operator or +"
and the other for "−". In every run, Minus is the only operator whose signal must move
the gate away from the state a number is in without it. The "trigger" defined above,
from the GLU effect, only reflects the readout sign. (Checked with a one-off loop over the
same 11 seeds, not part of the script's output.)

**Per-path shrinking.** This tests how the Minus signal reaches the gate. Minus can
affect a number's gate only through attention: through the queries and keys, it sets how
much the number attends to it (routing), and through the values, it sets what is copied
(content). Its own GLU update is zero, because its D0 is 0 and the value branch has no
bias.
- **Routing only (Q/K):** attention from a number to the preceding Minus falls (for
  example 0.47 → 0.36). OOD MAE rises moderately (120–980 in the 8 runs that break), and
  the flip ratio stays ≥ 0.97.
- **Content only (V and residual):** attention is unchanged. OOD MAE rises to
  2,400–8,700 in 6 runs, with flip ratio 0.35–0.86.
- **Both:** the effect is much larger than either alone (8,600–11,600, flip 0.04–0.36),
  because the signal at the number is roughly (attention weight) × (copied vector), and
  shrinking Minus reduces both factors.

Attention from a number to the Minus before it is only 0.47–0.77 in the unmodified
models. A long Minus vector compensates for this diffuse attention.

**Explanation, consistent with all of the above.** The gate needs a large enough Minus
signal at the number to switch, and that signal is a product of a partial attention weight
and the copied Minus vector. A long Minus embedding makes the product large enough. Plus
does not have to switch anything, since a number without any operator already has the
"+" gate state, so its length is free. This has not been tested directly: measuring the
gate's pre-activation margin with and without shrinking would test it.
