# ladder.py — rationale and results

## Why a factorial and not a ladder

The reviewer suggestion (via Fable) was a ladder from Config A to STRAT-Lite adding one
constraint per rung. A ladder's answer depends on the rung order: a constraint that only
helps once another is in place looks decisive in one order and useless in another. With
five binary factors the full factorial is 32 cells; at 50 seeds that is 1,600 runs
(≈ 12 min on 10 cores, 2026-11-14), so there was no reason to pick an order.

## Factors

Taken from a line-by-line comparison of `ablation.DenseBlock` and `lite/strat_lite.py`;
these five are every difference between the two.

| Factor | Constraint |
|---|---|
| G | gate reads Type dims (D1–D4) only |
| Q | Q/K read Type dims only; scores × 1/√4 (STRAT-Lite's scaling) |
| V | V reads Type dims only; attention writes Type dims only |
| F | value branch reads Data dims (D0, D5) only, no bias; GLU writes Data dims only |
| I | STRAT-Lite initialisation (gate bias 1.0, gate Xavier gain 0.5, value N(0, 0.01), readout Xavier) |

**Masks, not smaller matrices.** Constraints are applied by masking inputs and outputs of
the full 6×6 layers, so every cell builds the same layers in the same order. That makes
the all-off cell `L------` bit-identical to Config A (checked: seeds 1000, 1001). The
all-on cell `L-GQVFI` has STRAT-Lite's architecture but not its exact parameterisation:
STRAT-Lite uses 4×4 and 2-unit layers, so initialisation draws differ. It is therefore
compared with STRAT-Lite by success rate, not bit for bit. Masked weights receive zero
gradient and stay at their initial values.

## Results (2026-11-14; seeds 1000–1049; success = OOD MAE < 100)

- **Success needs G, Q and V together.** The four cells with G+Q+V: 167/200 = 83.5%
  (95% Wilson CI 77.7–88.0); the all-on cell 82% vs STRAT-Lite's 42/50 = 84%. The other
  28 cells: 40/1400 = 2.9% (CI 2.1–3.9).
- **F and I do not matter**: success 12.8% vs 13.1% with F on vs off, 11.8% vs 14.1% for I.
- **Pairs fail in ways that match the earlier measurements:**
  - Q+V without G (clean routing, gate still reads D0): 0% success, OOD MAE ≈ 11,400,
    OOD flip ratio 0, ID slope 1.0. This reproduces Bfa/Cb (oracle routing, gate reads
    D0): Magnitude Leak, now produced by training.
  - G+Q without V (gate barred from D0, but attention copies D0 into the Type dims the
    gate reads): 0% success, OOD MAE ≈ 11,500. The indirect path is enough to leak.
  - G alone or G+V (Q off): operator use survives OOD (flip ratio ≈ 1.0) but OOD MAE stays
    ≈ 5,500. The sign is handled correctly, yet the sum is still wrong, because attention
    weights that depend on magnitude mix values incorrectly at OOD scale.
  - Q alone: 24% success (6–24% across I/F). Clean attention weights sometimes suffice,
    presumably when training drives the gate's D0 weight towards zero; not checked.

## Relation to earlier findings

- The soft orthogonality penalty (Bortho) approximates G alone, and G alone gives 0–2%
  success, consistent with Bortho's failure.
- `attention_test.py` found that removing magnitude from Q/K did not rescue *trained*
  dense models. The factorial shows that Q/K isolation matters once the gate paths are
  closed: in trained Config A, the gate leak masks the attention effect. The two results
  are not in conflict, but the earlier one should not be read as "attention is irrelevant".
