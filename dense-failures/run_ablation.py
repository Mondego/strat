"""
Runs the signed-sum ablation (paper Table 2): each configuration x each seed, in parallel,
then prints per-configuration quantiles in the Table 2 layout.

Usage:
    python run_ablation.py                                  # all 13 configs, 50 seeds
    python run_ablation.py --configs A Bfl --seeds 5 --workers 4
Options: --configs, --seeds (default 50), --seed-offset (1000), --steps (2000),
         --workers (CPU count), --out (results/ablation next to this file).

Reads nothing. Writes <out>/results.csv (one row per run) and
<out>/checkpoints/<config>_<seed>.pt (trained state_dict of each run).

See run_ablation.py.md.
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

import ablation

HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(task):
    name, seed, steps, ckpt_dir = task
    torch.set_num_threads(1)
    t0 = time.time()
    model, train_mse, _ = ablation.train(name, seed, steps=steps)
    id_mae, id_solved = ablation.evaluate(model, name, *ablation.eval_set('id'))
    ood_mae, ood_solved = ablation.evaluate(model, name, *ablation.eval_set('ood'))
    torch.save(model.state_dict(), os.path.join(ckpt_dir, f'{name}_{seed}.pt'))
    return {'Config': name, 'Seed': seed, 'Steps': steps, 'Train_MSE': train_mse,
            'ID_MAE': id_mae, 'ID_Solved_Pct': id_solved,
            'OOD_MAE': ood_mae, 'OOD_Solved_Pct': ood_solved,
            'Duration_Sec': time.time() - t0}


def table2(df):
    """Per-config ID 25/50/75 and OOD 50/75 percentiles of MAE, in CONFIGS order."""
    rows = []
    for name in [c for c in ablation.CONFIGS if c in set(df.Config)]:
        g = df[df.Config == name]
        rows.append({'Config': name, 'n': len(g),
                     'ID_25': np.percentile(g.ID_MAE, 25), 'ID_50': g.ID_MAE.median(),
                     'ID_75': np.percentile(g.ID_MAE, 75),
                     'OOD_50': g.OOD_MAE.median(), 'OOD_75': np.percentile(g.OOD_MAE, 75)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', default=list(ablation.CONFIGS))
    ap.add_argument('--seeds', type=int, default=50)
    ap.add_argument('--seed-offset', type=int, default=1000)
    ap.add_argument('--steps', type=int, default=ablation.STEPS)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'ablation'))
    args = ap.parse_args()

    unknown = set(args.configs) - set(ablation.CONFIGS)
    if unknown:
        ap.error(f'unknown configs {sorted(unknown)}; choose from {list(ablation.CONFIGS)}')
    ckpt_dir = os.path.join(args.out, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    tasks = [(name, args.seed_offset + i, args.steps, ckpt_dir)
             for i in range(args.seeds) for name in args.configs]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, row in enumerate(pool.map(run_one, tasks), 1):
            rows.append(row)
            print(f"[{k}/{len(tasks)}] {row['Config']:<10} seed {row['Seed']}  "
                  f"ID {row['ID_MAE']:9.2f}  OOD {row['OOD_MAE']:10.1f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, 'results.csv'), index=False)
    pd.set_option('display.width', 120)
    print('\nMAE percentiles (Table 2 layout):')
    print(table2(df).to_string(index=False, float_format=lambda v: f'{v:.1f}'))


if __name__ == '__main__':
    main()
