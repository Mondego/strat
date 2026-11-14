# run_ablation.py — rationale

## What it replaces

`driver.py`, which re-implemented training per config and diverged from the per-file
scripts (see `ablation.py.md`). This runner contains no training logic; it only fans
`ablation.train` / `ablation.evaluate` out over (config, seed) pairs.

## Choices

- **Seeds 1000–1049** (`--seed-offset 1000`, `--seeds 50`): same seed range as the old
  driver, so runs of unchanged configurations are comparable with earlier outputs.
- **Parallel with one torch thread per worker.** The models are 6-dimensional; intra-op
  threading only adds contention. Each run seeds itself and draws its data from its own
  `random.Random(seed)`, so the results do not depend on worker count or scheduling.
- **Checkpoints for every run** (`checkpoints/<config>_<seed>.pt`, a few KB each), so the
  mechanism probes (gate statistics, flip tests, attention routing) can be computed later
  from saved weights without retraining.
- **Table 2 layout** in the printed summary: ID MAE 25th/50th/75th percentiles and OOD
  MAE median/75th percentile, as in the paper.

## Runtime

A 2000-step run takes about 4 s on one core of the author's laptop (Apple Silicon,
measured 2026-11-14); 13 configs × 50 seeds on 10 workers takes a few minutes.
