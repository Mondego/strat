# ablation.py — rationale

## Why one module

Until 2026-11 the ablation lived in `learn_ConfigA.py` … `learn_ConfigL.py` (each with
its own training loop) plus a `driver.py` that re-implemented a generic loop. The driver
silently dropped per-config training behaviour:

- `ConfigL` (Acurr): the curriculum existed only in the file's own loop. Under the
  driver, Acurr was bit-for-bit identical to Config A (verified 2026-11-14, seeds 1000–1001).
- `ConfigI` (As): the driver scaled targets by 0.01 but left inputs raw.
- `ConfigJ` (Aln): the driver gave it lr 0.005 via a comment meant for "VATS12"
  (an old name for STRAT), while STRAT-Lite ran at 0.01.

Each file also hard-coded `torch.manual_seed(42)` and never seeded Python's `random`, so
running a file repeatedly gave identical initial weights with different data.

Now every configuration goes through the same `train()` / `evaluate()`; what differs
between configurations is declared in `CONFIGS` and nowhere else.

## Provenance of the paper's Table 2

Table 2 cannot be reproduced from any committed code. Running each old file's own loop
for seeds 1000–1049 (2026-11-14) gave numbers close to the paper for Bfl, Bfa, Cb, Aln
and Acurr; in range but with roughly double the median ID MAE for A, Bortho and Cbi
(≈32 vs 10–16); and nothing like the paper for As (old ÷100 scaling: median ID MAE 4.1
vs 98.7 in the paper). Most likely Table 2 came from an earlier, uncommitted version of
the per-file scripts. The rerun with this module replaces it.

## Equivalence with the old code

For configurations whose definition did not change (A, Bfl, Bfa, Cb, Bortho, Cc, Aln at
lr 0.01, Cbi, STRAT-Lite), `train(name, seed, steps=2001)` reproduces the old driver
bit-for-bit: same final training MSE, ID MAE and OOD MAE on seeds 1000 and 1001
(checked 2026-11-14). That relies on `DenseBlock` creating its layers in the old order
(W_q, W_k, W_v, ffn.w_gate, ffn.w_val, W_out), which fixes how initialisation consumes
torch's RNG, and on `random.Random(seed)` producing the same stream as `random.seed(seed)`.

## Decisions (2026-11-14)

- **Paper is the source of truth for config definitions.** Where code and Appendix B
  disagreed, the code was changed:
  - **As**: inputs (D0) and targets divided by **5000**, outputs multiplied back (paper:
    γ = 1/5000). The old file used 100, which leaves OOD inputs at 20–50, so the test
    was not scale-matched.
  - **Cg**: learnable scalar gain on the attention output, **init 1.0**, everything else
    learnable (paper B.1). The old `ConfigD` was a different experiment (frozen logic +
    learnable gain init 5) and has been dropped.
  - **Csb**: **fixed** gain γ = 100 (a buffer, not trained), everything else learnable.
    The old `ConfigF` used a learnable gain init 50 and has been dropped.
  - **Acurr**: curriculum max value 100 for the first 25% of steps, 1000 for the next
    25%, 5000 for the rest (paper B.1; boundaries 500/1000 at 2000 steps).
- **lr 0.01 for every configuration, STRAT-Lite included.** Every original per-file loop
  used 0.01; the 0.005 in the old driver applied to the wrong config.
- **2000 training steps** (paper: 2,000 steps × 64 = 128k samples). The old loops ran
  `range(2001)`; pass `steps=2001` to reproduce them.
- **Fixed evaluation sets.** ID (1–100) and OOD (2000–5000), 1000 sequences each, drawn
  once from `EVAL_SEED = 12345` (+1 for OOD) and shared by every configuration and seed.
  The old code drew evaluation data from the continuing training stream, so each run was
  scored on different sequences.

## Constants carried over unchanged

- Orthogonality penalty λ = 10 on Σ W_gate[:, D0]² (Bortho, Cc), as in the paper and old code.
- Oracle attention handshake gain 10, frozen-logic gate gain 100 and bias −150 (Appendix A).
- Cb and Cc: learnable gain init 50. Cbi: gate bias init +1.0.
- Gradient clipping at global norm 1.0, Adam, MSE, batch 64.

## Known inconsistencies

- **Cc definition.** Table 9 says "Config Bortho + Config Cb" (which would include frozen
  oracle attention). Appendix B.1 says Csb + Bortho with a learnable gain init 50 and
  learned attention. The code follows B.1. Table 2 also calls this row "Bc".
- **Padding leaks into the output.** Sequences are right-padded with zero vectors and the
  readout sums over every position. In `DenseBlock`, pad positions attend (causally) to
  real tokens and pass through the GLU biases, so they contribute to the sum. A
  sequence's prediction therefore depends on how much padding its batch has. STRAT-Lite
  is much less exposed: its value branch has no bias and pads carry no data. This is
  inherited from the original code and not fixed.
  Measured 2026-11-14 on the 50-seed rerun by re-evaluating each checkpoint with the
  sequences grouped by length (no padding). Median OOD MAE moves by under 4% for every
  configuration except Acurr (2526 padded → 3008 unpadded); STRAT-Lite is unchanged (40).
  So padding does not explain the dense-vs-STRAT OOD gap. One outlier: **As** has median
  ID MAE 42 padded but 182 unpadded, so the scaled model leans on pad contributions
  in-distribution.
