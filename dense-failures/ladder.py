"""
Constraint ladder from Config A to STRAT-Lite, run as a full factorial: which of
STRAT-Lite's architectural constraints turns OOD failure into success?

Five binary factors, each one of STRAT-Lite's differences from Config A (DenseBlock):
  G  the GLU gate reads Type dimensions (D1-D4) only
  Q  Q/K read Type dimensions only; scores scaled by 1/sqrt(4) as in STRAT-Lite
  V  attention V reads Type dimensions only, and attention writes Type dimensions only
  F  the GLU value branch reads Data dimensions (D0, D5) only, has no bias, and the GLU
     writes Data dimensions only
  I  STRAT-Lite initialisation: gate bias 1.0, gate weights Xavier (gain 0.5), value
     weights N(0, 0.01), readout Xavier
Cells are named L-GQVFI with '-' for an absent factor: 'L------' is Config A, 'L-GQVFI'
is the STRAT-Lite architecture.

Usage:
    python ladder.py --check                     # 'L------' must reproduce Config A exactly
    python ladder.py [--seeds 50] [--workers N] [--out results/ladder]
Writes <out>/ladder.csv (one row per run) and <out>/checkpoints/<cell>_<seed>.pt, and
prints success rates per cell and per factor.

See ladder.py.md.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import ablation
from attention_test import flip_batch

HERE = os.path.dirname(os.path.abspath(__file__))
FACTORS = 'GQVFI'
TYPE = torch.tensor([0., 1., 1., 1., 1., 0.])
DATA = torch.tensor([1., 0., 0., 0., 0., 1.])
SUCCESS_OOD_MAE = 100.0


class LadderBlock(ablation.DenseBlock):
    """DenseBlock with any subset of STRAT-Lite's constraints, applied as input/output masks."""

    def __init__(self, G=False, Q=False, V=False, F=False, I=False):
        super().__init__()                # same layers, same creation order as Config A
        self.G, self.Q, self.V, self.F = G, Q, V, F
        if F:
            with torch.no_grad():
                self.ffn.w_val.bias.zero_()
            self.ffn.w_val.bias.requires_grad = False
        if I:
            with torch.no_grad():
                self.ffn.w_gate.bias.fill_(1.0)
            nn.init.xavier_uniform_(self.ffn.w_gate.weight, gain=0.5)
            nn.init.normal_(self.ffn.w_val.weight, std=0.01)
            nn.init.xavier_uniform_(self.W_out.weight)

    def forward(self, x):
        T = x.shape[1]
        hq = x * TYPE if self.Q else x
        scores = self.W_q(hq) @ self.W_k(hq).transpose(-2, -1)
        if self.Q:
            scores = scores / 2.0
        idx = torch.arange(T, device=x.device)
        scores = scores - (idx.view(-1, 1) - idx.view(1, -1)).abs().float()
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn = torch.softmax(scores.masked_fill(causal, float('-inf')), dim=-1)
        self.last_attn = attn.detach()
        Z = attn @ self.W_v(x * TYPE if self.V else x)
        if self.V:
            Z = Z * TYPE
        combined = x + Z
        gate = torch.sigmoid(self.ffn.w_gate(combined * TYPE if self.G else combined))
        update = gate * self.ffn.w_val(combined * DATA if self.F else combined)
        if self.F:
            update = update * DATA
        return self.W_out((combined + update).sum(dim=1)).squeeze(-1)


def cell_name(bits):
    return 'L-' + ''.join(f if b else '-' for f, b in zip(FACTORS, bits))


CELLS = [cell_name(bits) for bits in product([0, 1], repeat=len(FACTORS))]
ablation.CONFIGS.update({
    cell_name(bits): ablation.Spec(partial(LadderBlock, **dict(zip(FACTORS, map(bool, bits)))))
    for bits in product([0, 1], repeat=len(FACTORS))})


def run_one(task):
    name, seed, ckpt_dir = task
    torch.set_num_threads(1)
    model, train_mse, _ = ablation.train(name, seed)
    row = {'Cell': name, 'Seed': seed, **{f: name[2 + i] == f for i, f in enumerate(FACTORS)},
           'Train_MSE': train_mse}
    for which in ('id', 'ood'):
        x, y = ablation.eval_set(which)
        pred = ablation.predict(model, name, x)
        x2, expected = flip_batch(x)
        dy = ablation.predict(model, name, x2) - pred
        row[f'{which}_mae'] = (pred - y).abs().mean().item()
        row[f'{which}_flip'] = (dy / expected).median().item()
        if which == 'id':
            v = (expected.abs() / 2).numpy()
            row['id_flip_slope'], row['id_flip_intercept'] = np.polyfit(
                v, (dy * expected.sign() / 2).numpy(), 1)
    row['Success'] = row['ood_mae'] < SUCCESS_OOD_MAE
    torch.save(model.state_dict(), os.path.join(ckpt_dir, f'{name}_{seed}.pt'))
    return row


def check():
    base = cell_name((0,) * len(FACTORS))
    ok = True
    for seed in (1000, 1001):
        a_model, a_mse, _ = ablation.train('A', seed)
        l_model, l_mse, _ = ablation.train(base, seed)
        x, y = ablation.eval_set('ood')
        a_ood = ablation.evaluate(a_model, 'A', x, y)[0]
        l_ood = ablation.evaluate(l_model, base, x, y)[0]
        same = (a_mse, a_ood) == (l_mse, l_ood)
        ok &= same
        print(f'seed {seed}: A ({a_mse:.4f}, {a_ood:.2f})  {base} ({l_mse:.4f}, {l_ood:.2f})  '
              f'{"IDENTICAL" if same else "DIFFERENT"}')
    print('CHECK PASSED' if ok else 'CHECK FAILED')
    return ok


def summarise(df):
    cells = df.groupby('Cell').agg(success_pct=('Success', 'mean'), ood_mae_median=('ood_mae', 'median'),
                                   ood_flip_median=('ood_flip', 'median'),
                                   id_slope_median=('id_flip_slope', 'median')).reindex(CELLS)
    cells['success_pct'] *= 100
    marg = pd.DataFrame({f: {'success_on': df[df[f]].Success.mean() * 100,
                             'success_off': df[~df[f]].Success.mean() * 100} for f in FACTORS}).T
    return cells, marg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--seeds', type=int, default=50)
    ap.add_argument('--seed-offset', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'ladder'))
    args = ap.parse_args()
    if args.check:
        raise SystemExit(0 if check() else 1)
    ckpt_dir = os.path.join(args.out, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    tasks = [(c, args.seed_offset + i, ckpt_dir) for i in range(args.seeds) for c in CELLS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        df = pd.DataFrame(list(pool.map(run_one, tasks)))
    df.to_csv(os.path.join(args.out, 'ladder.csv'), index=False)
    cells, marg = summarise(df)
    pd.set_option('display.width', 160)
    print(cells.round(2).to_string())
    print('\nSuccess % with each factor on vs off (averaged over the other factors):')
    print(marg.round(1).to_string())


if __name__ == '__main__':
    main()
