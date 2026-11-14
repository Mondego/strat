"""
Exploratory, post-hoc test: does magnitude reaching the attention scores (Q/K) explain why
dense models use the operators in distribution but ignore them out of distribution?

For each trained run, re-evaluates under four conditions in which the operands seen at
one site are replaced by the ID mean (50.5), while every other computation uses the real
operands: none; qk (inputs to W_q and W_k); gate (input to the GLU gate); qk+gate.
Reports MAE and the operator flip ratio (1 = operator used correctly, 0 = ignored).
Also regresses the in-distribution flip effect on the flipped operand v: a correct model's
effect is proportional to v (slope 1, intercept 0); a model that learned the operator as a
fixed offset has slope ~0 and intercept ~ the typical training operand.

Usage:
    python attention_test.py --selftest
    python attention_test.py [--results results/ablation] [--workers N]
Reads <results>/checkpoints/*.pt. Writes <results>/attention_test.csv (one row per run)
and prints per-configuration medians. --selftest writes nothing.

See attention_test.py.md.
"""
import argparse
import glob
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

import ablation
from probes import FLIP_SEED, ID_MEAN, forward, handbuilt

HERE = os.path.dirname(os.path.abspath(__file__))
SITES = {'qk': ('W_q', 'W_k'), 'gate': ('ffn.w_gate',)}
CONDITIONS = {'none': (), 'qk': ('qk',), 'gate': ('gate',), 'qk+gate': ('qk', 'gate')}


class Substituter:
    """Pre-hooks that capture a site's input on a mean-operand pass and replay it."""

    def __init__(self, model):
        self.model, self.mode, self.active, self.store = model, None, (), {}
        for site, modules in SITES.items():
            for name in modules:
                model.get_submodule(name).register_forward_pre_hook(self._hook(site, name))

    def _hook(self, site, name):
        def hook(module, args):
            if site not in self.active:
                return None
            if self.mode == 'capture':
                self.store[name] = args[0].detach()
                return None
            return (self.store[name],)
        return hook

    def predict(self, name, x, sites):
        with torch.no_grad():
            self.active = sites
            if sites:
                num = x[..., 1] == 1
                xm = x.clone()
                xm[..., 0] = torch.where(num, torch.full_like(xm[..., 0], ID_MEAN), xm[..., 0])
                self.mode = 'capture'
                forward(self.model, name, xm)
                self.mode = 'replace'
            out = forward(self.model, name, x)
            self.active, self.mode = (), None
        return out


def flip_batch(x):
    """Same operator flips as probes.py (seed 777): flipped batch and expected change."""
    rng = random.Random(FLIP_SEED)
    x2, expected = x.clone(), torch.empty(x.shape[0])
    for b in range(x.shape[0]):
        k = rng.choice((x[b, :, 4] == 1).nonzero().squeeze(1).tolist())
        was_plus = x[b, k, 2] == 1
        x2[b, k, 2], x2[b, k, 3] = x[b, k, 3], x[b, k, 2]
        expected[b] = (-2.0 if was_plus else 2.0) * x[b, k + 1, 0]
    return x2, expected


def test_model(model, name):
    model.eval()
    sub = Substituter(model)
    row = {}
    for which in ('id', 'ood'):
        x, y = ablation.eval_set(which)
        x2, expected = flip_batch(x)
        for cond, sites in CONDITIONS.items():
            pred = sub.predict(name, x, sites)
            pred2 = sub.predict(name, x2, sites)
            row[f'{which}_mae_{cond}'] = (pred - y).abs().mean().item()
            row[f'{which}_flip_{cond}'] = ((pred2 - pred) / expected).median().item()
            if which == 'id' and cond == 'none':
                v = (expected.abs() / 2).numpy()
                effect = ((pred2 - pred) * expected.sign() / 2).numpy()   # = v when correct
                row['id_flip_slope'], row['id_flip_intercept'] = np.polyfit(v, effect, 1)
    return row


def test_checkpoint(path):
    torch.set_num_threads(1)
    name, seed = os.path.basename(path)[:-3].rsplit('_', 1)
    model = ablation.CONFIGS[name].make_model()
    model.load_state_dict(torch.load(path))
    return {'Config': name, 'Seed': int(seed), **test_model(model, name)}


def attention_leak_model():
    """Appendix A ALU whose number queries weaken with magnitude: routing fails at OOD scale."""
    m = handbuilt('correct')
    with torch.no_grad():
        m.W_q.weight[4, 0] = -0.01
    return m


SELFTEST_EXPECT = {   # model: {column: (low, high)}
    'correct':        {'ood_mae_none': (0, 1e-3), 'ood_mae_qk': (0, 1e-3), 'ood_mae_gate': (0, 1e-3),
                       'ood_flip_none': (.99, 1.01), 'id_flip_slope': (.99, 1.01),
                       'id_flip_intercept': (-.01, .01)},
    'offset':         {'id_flip_slope': (-.01, .01), 'id_flip_intercept': (49.4, 51.6)},
    'gate_leak':      {'ood_flip_none': (-.01, .01), 'ood_flip_qk': (-.01, .01),
                       'ood_flip_gate': (.99, 1.01), 'ood_mae_gate': (0, 1e-3)},
    'attention_leak': {'id_mae_none': (0, 1e-3), 'ood_flip_none': (-.01, .01),
                       'ood_flip_qk': (.99, 1.01), 'ood_mae_qk': (0, 1e-3)},
}


class OffsetModel(torch.nn.Module):
    """Sum of operands, with each minus subtracting a fixed 2 x 50.5 instead of 2 x v."""

    def __init__(self):
        super().__init__()
        self.ffn, self.W_out = handbuilt('correct').ffn, None   # hooks need these modules
        self.W_q = self.W_k = torch.nn.Identity()

    def forward(self, x):
        num = x[..., 1] == 1
        prev_sub = torch.zeros_like(num)
        prev_sub[:, 1:] = x[:, :-1, 3] == 1
        return (x[..., 0] * num).sum(1) - 2 * ID_MEAN * (num & prev_sub).sum(1)


def selftest():
    models = {'correct': handbuilt('correct'), 'offset': OffsetModel(),
              'gate_leak': handbuilt('magnitude_leak'),
              'attention_leak': attention_leak_model()}
    ok = True
    for key, expect in SELFTEST_EXPECT.items():
        row = test_model(models[key], 'A')
        for col, (lo, hi) in expect.items():
            good = lo <= row[col] <= hi
            ok &= good
            print(f"{key:<15} {col:<16} {row[col]:12.4f}  expect [{lo}, {hi}]  {'ok' if good else 'FAIL'}")
    print('SELFTEST PASSED' if ok else 'SELFTEST FAILED')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--results', default=os.path.join(HERE, 'results', 'ablation'))
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    paths = sorted(glob.glob(os.path.join(args.results, 'checkpoints', '*.pt')))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        df = pd.DataFrame(list(pool.map(test_checkpoint, paths))).sort_values(['Config', 'Seed'])
    df.to_csv(os.path.join(args.results, 'attention_test.csv'), index=False)

    order = [c for c in ablation.CONFIGS if c in set(df.Config)]
    cols = [f'ood_{m}_{c}' for m in ('flip', 'mae') for c in CONDITIONS] + \
           [f'id_mae_{c}' for c in CONDITIONS] + ['id_flip_slope', 'id_flip_intercept']
    pd.set_option('display.width', 250)
    print('Per-configuration medians (OOD flip: 1 = operator used, 0 = ignored):')
    print(df.groupby('Config')[cols].median().reindex(order).round(2).to_string())


if __name__ == '__main__':
    main()
