# attention_test.py — rationale

**Exploratory and post hoc.** This test was designed after the first labelling run
(2026-11-14), to explain a pattern the pre-registered rules left Unclassified: 79% of
dense runs use the operators correctly in distribution (flip ratio ≈ 1) and ignore them
out of distribution (≈ 0). It must be presented as exploratory in the rebuttal.

## Design

**Mean-operand substitution.** At one site (Q/K inputs, or the GLU gate input), the
input is replaced by what that site would receive if every operand were 50.5 (the ID
mean). Everything else sees the real operands. The replacement comes from a separate
forward pass on the mean-operand copy, so it includes everything upstream of the site.

**Why the gate ablation already covers attention-into-gate.** The gate's replacement
input is taken from a full pass in which attention also saw mean operands. "Magnitude
disturbs attention, which then disturbs the gate" is therefore removed by the gate
condition alone. The Q/K condition isolates a different path: magnitude-dependent
attention *weights* applied to real values in the residual and value branch.

**Flip regression.** Once attention weights and gates are pinned, the rest of a dense
block is linear in the operands, so an operator flip can only change the output by
a·v + b. A correct model has a = 1 (in units of 2v) and b = 0. A model that encodes
"minus" as a fixed amount has a ≈ 0 and b ≈ the typical training operand (≈ 50). This
distinguishes the two parts in distribution, where the median flip ratio cannot (an
offset of 2 × 50 gives a ratio ≈ 1 at v ≈ 50).

## Validation (`--selftest`, 2026-11-14)

All expectations met: the Appendix A ALU is unchanged by every ablation; a gate that
reads magnitude is fixed by the gate condition and not by Q/K; a query that weakens with
magnitude (W_q[4, 0] = −0.01, so routing fails at OOD scale) is fixed by Q/K; a pure
offset model gives slope 0 and intercept 50.5.

**The gate ablation also fixes the attention-leak model** (checked 2026-11-18, now in the
self-test). Its first pass records the gate input with every operand at 50.5, and that
input includes the flags attention copies to the number; with operands at 50.5 the
routing works, so the gate receives the correct flags. The two ablations are therefore not
symmetric: Q/K repairs only the attention-leak model, while the gate ablation repairs both.
A recovery under the gate ablation means operand magnitude reaches the gate input, directly
or through attention (weights or values); only the Q/K ablation isolates the attention
weights. An earlier version of the paper (B.4) said each ablation repaired only the model
whose path it targets; that was not tested and is false for the gate ablation.

## Results (2026-11-14, 650 runs; medians per config)

- **Q/K ablation changes nothing.** OOD flip ratio stays at 0.00–0.06 in every dense config.
  Magnitude in the attention scores is not why operator use fails OOD.
- **Gate ablation**: restores OOD operator use fully in Bfa and Cb (flip 0.99–1.00; OOD
  MAE 11,833 → 989 and 11,537 → 156); partly in A, Bortho, Cg, Cbi (flip 0.29–0.37);
  not at all in As, Aln, Csb, Cc, Bfl.
- **Flip regression in distribution** (slope, intercept):
  Bfa, Cb, STRAT-Lite (1.00, ≈ 0): correct form; Csb, Cc, Bfl (0.01–0.05, 29–49): pure
  offset; A, Bortho, Cg, Cbi (0.40–0.49, 22–29): half and half; As (0.30, 33);
  Aln (0.77, 12).

**Reading.** In 8 of the 12 dense configs the OOD failure decomposes into (i) a
proportional part computed through a gate that also reads magnitude (restored by the
gate ablation) and (ii) an offset part calibrated to the training magnitude (never
restorable, by construction). For A, Bortho, Cg and Cbi, the restored OOD flip ratio
(0.29–0.37) is close to the in-distribution slope (0.40–0.49).

**Not explained.** As and Aln have a proportional part (slope 0.30 and 0.77) that the gate
ablation does not restore, so their failure runs through another path. A plausible one
for Aln is LayerNorm on the residual shrinking the operator flags at large magnitude
(untested).

**Per-run correlation (corrected 2026-11-18).** An earlier version reported that slope and
gate-restored OOD flip correlate only at r = 0.37 across individual dense runs, and
concluded that the decomposition holds for configurations, not runs. That r included
Acurr, whose slope and ablation are not comparable (slope −0.04, restored flip 0.62), and
it drove the value down. Over the 200 runs of A, Bortho, Cg and Cbi, r = 0.61 (Spearman
0.63), and 0.52–0.69 within each configuration; over the 550 dense runs without Acurr,
r = 0.68. The restored flip is below the slope in 70% of those 200 runs (median
difference −0.10). The decomposition therefore also holds, less tightly, per run.

**Acurr caveat.** Acurr trains on operands up to 5000, so 50.5 is not its typical operand
and its ablations are not comparable. Its in-distribution slope (−0.04, intercept 273)
reflects a model fitted to large operands, evaluated on small ones.
