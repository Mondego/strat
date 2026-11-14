"""
Retrains ablation runs with training-time logging: gradient norms per parameter group
(before clipping) and gate saturation at number tokens. Evidence for or against the
"Gradient Wall", which cannot be measured from final weights.

Usage: python dynamics.py [--configs A Bfl ...] [--seeds 50] [--seed-offset 1000]
                          [--every 50] [--workers N] [--results results/ablation]
Reads <results>/results.csv to check that logging leaves training unchanged (final training
MSE must match exactly). Writes <results>/dynamics.csv: one row per (config, seed, logged step).

See dynamics.py.md.
"""
import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import torch

import ablation
from probes import SAT_HI, SAT_LO

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = {'attn': ('W_q.', 'W_k.', 'W_v.'), 'gate': ('ffn.w_gate.',), 'val': ('ffn.w_val.',),
          'out': ('W_out.',), 'gain': ('attn_gain',)}


def run_one(task):
    name, seed, every = task
    torch.set_num_threads(1)
    rows = []

    def on_step(step, model, x_in, mse):
        if step % every and step != ablation.STEPS - 1:
            return
        row = {'Config': name, 'Seed': seed, 'Step': step, 'MSE': mse}
        params = dict(model.named_parameters())
        for group, prefixes in GROUPS.items():
            grads = [p.grad for n, p in params.items() if n.startswith(prefixes) and p.grad is not None]
            row[f'grad_{group}'] = (math.sqrt(sum(g.pow(2).sum().item() for g in grads))
                                    if grads else float('nan'))
        # Gate activations on this batch: an extra no-grad forward (deterministic, no RNG).
        logits = {}
        h = model.ffn.w_gate.register_forward_hook(lambda m, a, o: logits.__setitem__('l', o))
        with torch.no_grad():
            model(x_in)
        h.remove()
        g = torch.sigmoid(logits['l'])[x_in[..., 1] == 1]
        row['gate_saturated'] = ((g < SAT_LO) | (g > SAT_HI)).float().mean().item()
        row['gate_mean'] = g.mean().item()
        rows.append(row)

    _, final_mse, _ = ablation.train(name, seed, on_step=on_step)
    return rows, (name, seed, final_mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', default=list(ablation.CONFIGS))
    ap.add_argument('--seeds', type=int, default=50)
    ap.add_argument('--seed-offset', type=int, default=1000)
    ap.add_argument('--every', type=int, default=50)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--results', default=os.path.join(HERE, 'results', 'ablation'))
    args = ap.parse_args()

    tasks = [(n, args.seed_offset + i, args.every) for i in range(args.seeds) for n in args.configs]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        out = list(pool.map(run_one, tasks))

    ref = pd.read_csv(os.path.join(args.results, 'results.csv'),
                      float_precision='round_trip').set_index(['Config', 'Seed']).Train_MSE
    mismatched = [(n, s) for _, (n, s, mse) in out if (n, s) in ref.index and ref[(n, s)] != mse]
    if mismatched:
        raise SystemExit(f'logging changed training for {mismatched[:5]} ... ({len(mismatched)} runs)')
    rows = [r for rs, _ in out for r in rs]
    path = os.path.join(args.results, 'dynamics.csv')
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'wrote {len(rows)} rows to {path}; final training MSE identical to results.csv '
          f'for all {len(out)} runs')


if __name__ == '__main__':
    main()
