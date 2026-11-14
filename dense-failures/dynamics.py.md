# dynamics.py — rationale

The "Gradient Wall" claims that gradients stop reaching attention because the gates
saturate. That is a statement about training, not about the final weights, so it needs
a logged training pass.

## Choices

- **Retrain instead of logging during the original run.** `run_ablation.py` stays free of
  diagnostics. Training is deterministic per seed, so the logged rerun is the same run:
  the script refuses to write output unless every run's final training MSE equals the
  value in `results.csv` exactly. Verified 2026-11-14 on seed 1000 for all 13 configs.
- **Norms before clipping.** Clipping is global at norm 1.0, so norms after clipping would
  hide which group is starved. `ablation.train` calls the hook between `backward()` and
  the clip.
- **Groups**: attention (W_q, W_k, W_v), gate (ffn.w_gate), value (ffn.w_val), readout
  (W_out), gain (attn_gain). Frozen groups have no gradient and are logged as NaN
  (attention in Bfa and Cb; GLU and readout in Bfl).
- **Saturation** uses the probe thresholds (σ < 0.05 or > 0.95), on number tokens of the
  current training batch, via an extra no-grad forward pass. That pass uses no RNG and
  does not touch gradients, which the final-MSE check confirms.
- **Every 50 steps**, plus the last step: 41 points per run, enough for a median over
  training without making the CSV large.

## Bug found while building it

pandas' default CSV float parser does not round-trip floats exactly, so the equality
check failed spuriously for As and STRAT-Lite (differences in the 16th digit). The check
reads `results.csv` with `float_precision='round_trip'`.
