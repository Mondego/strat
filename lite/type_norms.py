"""
Do the different norms of the learned Plus and Minus type embeddings matter?

Trains STRAT-Lite with learned types (STRAT12e from strat_lite_learned_types.py, same
training as its __main__: Adam 0.01, 2001 steps of fresh batches of 64, clip 1.0) for
seeds 1000-1009, plus seed 42 to compare with Table 3 of the paper. For each run:

  Test 1 (descriptive): norms of the Literal/Plus/Minus embeddings, cosine(Plus, Minus),
    sign of the readout weight on D0, and which operator triggers the GLU: the operator
    after which the GLU's contribution to the output, divided by the operand, is larger.
  Test 2 (intervention): at evaluation only, rescale one operator embedding (direction
    kept) so both have the same norm, in each direction (longer -> shorter norm, shorter
    -> longer norm). The shrinking is also applied to one attention path at a time: only
    the Q/K inputs (routing), or only the V input and residual (content). Reports ID/OOD
    MAE, the median OOD flip ratio, and the attention from a number to the operator
    before it.

Evaluation uses the fixed ID/OOD sets of dense-failures/ablation.py (converted to token
ids and values) and the operator flips of probes.py (seed 777).

Usage: python type_norms.py [--workers N] [--out ../dense-failures/results/type_norms.csv]
Writes one row per (seed, condition) to --out and prints a summary.

See type_norms.py.md.
"""
import argparse
import copy
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, '..', 'dense-failures'))
import ablation                                           # noqa: E402
from strat_lite_learned_types import STRAT12e, generate_batch_v2   # noqa: E402

SEEDS = [42] + list(range(1000, 1010))
FLIP_SEED = 777
LIT, PLUS, MINUS = 0, 1, 2


def to_ids(x):
    """ablation.py 6-dim input -> (token ids, values). Padding becomes Literal 0.0, as in training."""
    ids = torch.zeros(x.shape[:2], dtype=torch.long)
    ids[x[..., 2] == 1] = PLUS
    ids[x[..., 3] == 1] = MINUS
    return ids, x[..., 0].clone()


def train(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    model = STRAT12e()
    opt = optim.Adam(model.parameters(), lr=0.01)
    for _ in range(2001):
        ids, vals, y = generate_batch_v2(batch_size=64)
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(ids, vals).unsqueeze(1), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval()


def flip_batch(ids, vals):
    rng = random.Random(FLIP_SEED)
    ids2, expected = ids.clone(), torch.empty(ids.shape[0])
    for b in range(ids.shape[0]):
        k = rng.choice(((ids[b] == PLUS) | (ids[b] == MINUS)).nonzero().squeeze(1).tolist())
        was_plus = ids[b, k] == PLUS
        ids2[b, k] = MINUS if was_plus else PLUS
        expected[b] = (-2.0 if was_plus else 2.0) * vals[b, k + 1]
    return ids2, expected


def glu_effect(model, ids, vals):
    """Mean GLU contribution to the output divided by the operand, after + and after -."""
    rec = {}
    h = model.ffn.register_forward_hook(lambda m, i, o: rec.__setitem__('u', o))
    with torch.no_grad():
        model(ids, vals)
    h.remove()
    w = model.W_out.weight[0, [0, 5]]
    c = (rec['u'] @ w)[:, 1:] / vals[:, 1:].clamp(min=1e-9)
    prev = ids[:, :-1]
    is_num = vals[:, 1:] > 0
    return c[is_num & (prev == PLUS)].mean().item(), c[is_num & (prev == MINUS)].mean().item()


def forward_paths(model, ids, vals, E_qk, E_v, rec=None):
    """STRAT12e.forward with separate type tables for the Q/K inputs and for the V input
    and residual. With E_qk = E_v = the model's table it equals model(ids, vals)."""
    t_qk, t_v = E_qk[ids], E_v[ids]
    T = ids.shape[1]
    scores = model.W_q(t_qk) @ model.W_k(t_qk).transpose(-2, -1) / math.sqrt(4)
    idx = torch.arange(T)
    scores = scores - (idx.view(-1, 1) - idx.view(1, -1)).abs().float()
    scores = scores.masked_fill(torch.triu(torch.ones(T, T), diagonal=1) == 1, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    if rec is not None:
        rec['attn'] = attn
    d0 = vals.unsqueeze(-1)
    x = torch.cat([d0, t_v + attn @ model.W_v(t_v), torch.zeros_like(d0)], dim=-1)
    upd = model.ffn(x)
    x = torch.cat([x[..., :1] + upd[..., :1], x[..., 1:5], x[..., 5:] + upd[..., 1:]], dim=-1)
    return model.W_out(x.sum(dim=1)).squeeze(-1)


def attn_to_prev(model, ids, vals, E_qk, E_v):
    """Mean attention from a number to the operator just before it, after + and after -."""
    rec = {}
    with torch.no_grad():
        forward_paths(model, ids, vals, E_qk, E_v, rec)
    a = torch.diagonal(rec['attn'], offset=-1, dim1=1, dim2=2)      # attn[b, t, t-1]
    num, prev = vals[:, 1:] > 0, ids[:, :-1]
    return a[num & (prev == PLUS)].mean().item(), a[num & (prev == MINUS)].mean().item()


def evaluate(fwd):
    out = {}
    for which in ('id', 'ood'):
        x, y = ablation.eval_set(which)
        ids, vals = to_ids(x)
        with torch.no_grad():
            pred = fwd(ids, vals)
            out[f'{which}_mae'] = (pred - y).abs().mean().item()
            if which == 'ood':
                ids2, expected = flip_batch(ids, vals)
                out['ood_flip'] = ((fwd(ids2, vals) - pred) / expected).median().item()
    return out


def run(seed):
    torch.set_num_threads(1)
    model = train(seed)
    E = model.embedding.type_embeddings.weight.detach()
    n = E.norm(dim=1)
    x, _ = ablation.eval_set('id')
    eff_plus, eff_minus = glu_effect(model, *to_ids(x))
    base = dict(seed=seed, norm_lit=n[LIT].item(), norm_plus=n[PLUS].item(),
                norm_minus=n[MINUS].item(),
                cos_pm=torch.nn.functional.cosine_similarity(E[PLUS], E[MINUS], dim=0).item(),
                readout_d0=model.W_out.weight[0, 0].item(),
                glu_after_plus=eff_plus, glu_after_minus=eff_minus,
                trigger='minus' if abs(eff_minus) > abs(eff_plus) else 'plus',
                emb_plus=E[PLUS].numpy().round(2).tolist(), emb_minus=E[MINUS].numpy().round(2).tolist())
    longer = PLUS if n[PLUS] > n[MINUS] else MINUS
    shorter = MINUS if longer == PLUS else PLUS
    shrunk, grown = E.clone(), E.clone()
    shrunk[longer] *= n[shorter] / n[longer]
    grown[shorter] *= n[longer] / n[shorter]
    conditions = {                       # name: (table for Q/K, table for V and residual)
        'none': (E, E),
        'longer_to_shorter': (shrunk, shrunk),
        'shorter_to_longer': (grown, grown),
        'longer_to_shorter_qk_only': (shrunk, E),
        'longer_to_shorter_v_only': (E, shrunk),
    }
    xi, _ = ablation.eval_set('id')
    rows = []
    for cond, (eqk, ev) in conditions.items():
        a_plus, a_minus = attn_to_prev(model, *to_ids(xi), eqk, ev)
        res = evaluate(lambda i, v: forward_paths(model, i, v, eqk, ev))
        rows.append({**base, 'condition': cond, 'attn_prev_plus': a_plus,
                     'attn_prev_minus': a_minus, **res})
    with torch.no_grad():                # forward_paths must reproduce the model exactly
        ids, vals = to_ids(xi)
        assert torch.allclose(forward_paths(model, ids, vals, E, E), model(ids, vals), atol=1e-3)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--out', default=os.path.join(HERE, '..', 'dense-failures', 'results', 'type_norms.csv'))
    args = ap.parse_args()
    with ProcessPoolExecutor(args.workers) as ex:
        rows = [r for rs in ex.map(run, SEEDS) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    pd.set_option('display.width', 250)
    b = df[df.condition == 'none']
    print(b[['seed', 'norm_lit', 'norm_plus', 'norm_minus', 'cos_pm', 'readout_d0',
             'glu_after_plus', 'glu_after_minus', 'trigger', 'ood_mae', 'ood_flip']].round(3).to_string(index=False))
    print(b[['seed', 'emb_plus', 'emb_minus']].to_string(index=False))
    for v in ('ood_mae', 'ood_flip', 'attn_prev_minus', 'attn_prev_plus'):
        print(f'\n{v}:')
        print(df.pivot_table(index='seed', columns='condition', values=v).round(2).to_string())


if __name__ == '__main__':
    main()
