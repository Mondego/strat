"""
Mechanism probes for trained ablation models: measures what each model does internally,
so failure labels can be assigned per seed instead of by interpretation.

Usage:
    python probes.py --selftest                  # check probes on hand-built models
    python probes.py [--results results/ablation] [--workers N]
Reads <results>/checkpoints/<config>_<seed>.pt and the fixed evaluation sets of ablation.py.
Writes <results>/probes.csv: one row per run, every probe on the ID and OOD sets
(columns prefixed id_ / ood_). --selftest writes nothing.

See probes.py.md for what each probe measures and why.
"""
import argparse
import glob
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

import ablation

HERE = os.path.dirname(os.path.abspath(__file__))
ID_MEAN = (ablation.ID_RANGE[0] + ablation.ID_RANGE[1]) / 2   # mean training operand, 50.5
FLIP_SEED = 777
SAT_LO, SAT_HI = 0.05, 0.95


# ------------------------------------------------------------------------- plumbing

class GLUProbe:
    """Hooks on the GLU: records gate input, gate logit, value branch; can override them."""

    def __init__(self, model):
        self.rec = {}
        self.input_override = None   # fn(gate_input) -> gate_input
        self.logit_override = None   # fn(gate_logit) -> gate_logit
        model.ffn.w_gate.register_forward_pre_hook(self._pre)
        model.ffn.w_gate.register_forward_hook(self._gate)
        model.ffn.w_val.register_forward_hook(self._val)

    def _pre(self, module, args):
        inp = args[0] if self.input_override is None else self.input_override(args[0])
        self.rec['gate_in'] = inp.detach()
        return (inp,)

    def _gate(self, module, args, out):
        if self.logit_override is not None:
            out = self.logit_override(out)
        self.rec['gate_logit'] = out.detach()
        return out

    def _val(self, module, args, out):
        self.rec['val'] = out.detach()


def write_dims(model):
    """Residual dimension each gate unit writes to (all 6 for DenseBlock, D0/D5 for STRAT)."""
    return list(model.ffn.idx_data) if hasattr(model.ffn, 'idx_data') else list(range(ablation.D))


def forward(model, name, x):
    """Prediction in raw units (handles the As input/output scaling); keeps autograd."""
    scale = ablation.CONFIGS[name].scale
    return model(ablation.scale_inputs(x, scale)).reshape(-1) * scale


def operand_signs(x):
    """Number-token mask and the sign each number carries (+1, or -1 after a minus)."""
    num = x[..., 1] == 1
    prev_sub = torch.zeros_like(num)
    prev_sub[:, 1:] = x[:, :-1, 3] == 1
    return num, torch.where(prev_sub, -1.0, 1.0)


def mae(pred, y):
    return (pred - y).abs().mean().item()


def corr2(a, b):
    a, b = a.double(), b.double()
    c = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    return c * c if math.isfinite(c) else float('nan')


# --------------------------------------------------------------------------- probes

def reference_stats(model, name, probe):
    """Gate statistics on ID number tokens, used as the replacement values for ablations."""
    x, _ = ablation.eval_set('id')
    num, _ = operand_signs(x)
    with torch.no_grad():
        forward(model, name, x)
    gate = torch.sigmoid(probe.rec['gate_logit'])
    gin = probe.rec['gate_in']
    unit_w = model.W_out.weight.detach()[0, write_dims(model)].abs() * probe.rec['val'][num].abs().mean(0)
    return {'gate_mean': gate[num].mean(0), 'unit_w': unit_w,
            'gin0_mean': gin[..., 0][num].mean() if gin.shape[-1] == ablation.D else None}


def probe_set(model, name, probe, x, y, ref):
    num, sign = operand_signs(x)
    plus, minus = num & (sign > 0), num & (sign < 0)
    r = {}

    # Base pass, with input gradients: per-operand sensitivity dy/dv.
    xg = x.clone().requires_grad_(True)
    pred = forward(model, name, xg)
    (grad,) = torch.autograd.grad(pred.sum(), xg)
    pred = pred.detach()
    sens = grad[..., 0]
    r['mae'] = mae(pred, y)
    r['sens_plus'] = sens[plus].mean().item()
    r['sens_minus'] = sens[minus].mean().item()
    r['sign_gap'] = r['sens_plus'] - r['sens_minus']

    # Gate activations at number tokens, each unit weighted by its path to the output.
    gate = torch.sigmoid(probe.rec['gate_logit'])
    w = ref['unit_w']
    geff = (gate * w).sum(-1) / w.sum().clamp_min(1e-12)
    r['gate_plus'] = geff[plus].mean().item()
    r['gate_minus'] = geff[minus].mean().item()
    r['gate_diff'] = r['gate_plus'] - r['gate_minus']
    g = gate[num]
    r['gate_saturated'] = ((g < SAT_LO) | (g > SAT_HI)).float().mean().item()

    # Routing: attention from each number to the token right before it (its operator).
    attn = getattr(model, 'last_attn', None)
    if attn is not None and x.shape[1] > 1:
        idx = torch.arange(1, x.shape[1])
        m = num[:, 1:]
        r['route_mass'] = attn[:, idx, idx - 1][m].mean().item()
        r['route_argmax'] = (attn[:, 1:, :].argmax(-1) == (idx - 1))[m].float().mean().item()

    with torch.no_grad():
        # Clamp: gate at number tokens fixed to its ID mean per unit.
        logit_mean = torch.logit(ref['gate_mean'].clamp(1e-6, 1 - 1e-6))
        probe.logit_override = lambda out: torch.where(num.unsqueeze(-1), logit_mean.expand_as(out), out)
        r['clamp_mae'] = mae(forward(model, name, x), y)
        probe.logit_override = None

        # Direct magnitude ablation: gate-input D0 at number tokens set to its ID mean.
        if ref['gin0_mean'] is not None:
            def direct(inp):
                inp = inp.clone()
                inp[..., 0] = torch.where(num, ref['gin0_mean'], inp[..., 0])
                return inp
            probe.input_override = direct
            r['d0_direct_mae'] = mae(forward(model, name, x), y)
            probe.input_override = None

        # Total magnitude ablation: the gate sees the input it would get if every operand
        # were the ID mean; the value branch still sees the real operands.
        xm = x.clone()
        xm[..., 0] = torch.where(num, torch.full_like(xm[..., 0], ID_MEAN), xm[..., 0])
        forward(model, name, xm)
        gin_m = probe.rec['gate_in'].clone()
        probe.input_override = lambda inp: gin_m
        r['d0_total_mae'] = mae(forward(model, name, x), y)
        probe.input_override = None

        # Operator flip: swap one operator per sequence, numbers fixed. Correct: dy = -2*s*v.
        rng = random.Random(FLIP_SEED)
        x2 = x.clone()
        expected = torch.empty(x.shape[0])
        for b in range(x.shape[0]):
            k = rng.choice((x[b, :, 4] == 1).nonzero().squeeze(1).tolist())
            was_plus = x[b, k, 2] == 1
            x2[b, k, 2], x2[b, k, 3] = x[b, k, 3], x[b, k, 2]
            expected[b] = (-2.0 if was_plus else 2.0) * x[b, k + 1, 0]
        r['flip_ratio'] = ((forward(model, name, x2) - pred) / expected).median().item()

        # Output vs. sign-blind and sign-aware sums.
        blind = (x[..., 0] * num).sum(1)
        r['r2_blind'] = corr2(pred, blind)
        r['r2_aware'] = corr2(pred, y)
        r['slope_on_target'] = (torch.cov(torch.stack([pred, y]))[0, 1] / y.var()).item()
        r['spread_ratio'] = (pred.std() / y.std()).item()
    return r


def probe_model(model, name):
    model.eval()
    probe = GLUProbe(model)
    ref = reference_stats(model, name, probe)
    row = {}
    for which in ('id', 'ood'):
        x, y = ablation.eval_set(which)
        for k, v in probe_set(model, name, probe, x, y, ref).items():
            row[f'{which}_{k}'] = v
    return row


def probe_checkpoint(path):
    torch.set_num_threads(1)
    name, seed = os.path.basename(path)[:-3].rsplit('_', 1)
    model = ablation.CONFIGS[name].make_model()
    model.load_state_dict(torch.load(path))
    return {'Config': name, 'Seed': int(seed), **probe_model(model, name)}


# ------------------------------------------------------------------------- selftest

def handbuilt(variant):
    """Appendix A ALU (oracle attention + frozen logic), optionally broken in a known way."""
    m = ablation.DenseBlock(oracle_attention=True, frozen_logic=True)
    with torch.no_grad():
        if variant == 'open_gate':
            m.ffn.w_gate.bias[5] = 1000.0        # gate always open
        elif variant == 'closed_gate':
            m.ffn.w_gate.bias[5] = -1000.0       # gate always closed: y = sum of |v|
        elif variant == 'magnitude_leak':
            m.ffn.w_gate.weight[5, 0] = -0.05    # large operands close the gate
    return m


SELFTEST_EXPECT = {   # variant: {probe column: (low, high)}
    'correct':        {'id_mae': (0, 1e-3), 'ood_mae': (0, 1e-3), 'id_sign_gap': (1.99, 2.01),
                       'ood_sign_gap': (1.99, 2.01), 'id_gate_plus': (0, .01), 'id_gate_minus': (.99, 1),
                       'id_flip_ratio': (.99, 1.01), 'id_route_argmax': (.99, 1), 'id_r2_aware': (.999, 1)},
    'open_gate':      {'id_gate_plus': (.99, 1), 'id_gate_minus': (.99, 1), 'id_sign_gap': (-.01, .01),
                       'id_flip_ratio': (-.01, .01), 'id_r2_blind': (.999, 1)},
    'closed_gate':    {'id_gate_plus': (0, .01), 'id_gate_minus': (0, .01), 'id_sign_gap': (-.01, .01),
                       'id_flip_ratio': (-.01, .01), 'id_r2_blind': (.999, 1)},
    'magnitude_leak': {'id_mae': (0, 1e-3), 'id_sign_gap': (1.99, 2.01), 'ood_sign_gap': (-.01, .01),
                       'ood_gate_minus': (0, .01), 'ood_d0_direct_mae': (0, 1e-3), 'ood_d0_total_mae': (0, 1e-3)},
}


def selftest():
    ok = True
    for variant, expect in SELFTEST_EXPECT.items():
        row = probe_model(handbuilt(variant), 'A')
        for col, (lo, hi) in expect.items():
            v = row[col]
            good = lo <= v <= hi
            ok &= good
            print(f"{variant:<15} {col:<20} {v:12.4f}  expect [{lo}, {hi}]  {'ok' if good else 'FAIL'}")
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
        rows = list(pool.map(probe_checkpoint, paths))
    df = pd.DataFrame(rows).sort_values(['Config', 'Seed'])
    df.to_csv(os.path.join(args.results, 'probes.csv'), index=False)
    print(f'wrote {len(df)} rows to {os.path.join(args.results, "probes.csv")}')


if __name__ == '__main__':
    main()
