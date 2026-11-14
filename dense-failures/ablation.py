"""
Signed-sum ablation (paper Section 4, Table 2, Appendix B): the dense 6-dim block, the
12 ablation configurations plus STRAT-Lite, data generation, training and evaluation.

Library module, imported by run_ablation.py. Reads nothing from disk, writes nothing.
Configuration names match the paper exactly (A, As, Aln, Acurr, Bfl, Bfa, Bortho, Cg,
Csb, Cc, Cb, Cbi, STRAT-Lite). Every configuration goes through the same train() and
evaluate(), so per-config behaviour lives only in CONFIGS.

See ablation.py.md.
"""
import os
import random
import sys
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

LITE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lite')
sys.path.insert(0, LITE)  # ahead of stdlib: lite/ has a test.py
from embeddings import get_tensor_embedding  # noqa: E402

D = 6                       # D0 value, D1 IsNum, D2 IsAdd, D3 IsSub, D4 IsOp, D5 scratch
ID_RANGE = (1, 100)
OOD_RANGE = (2000, 5000)
EVAL_SEED = 12345
EVAL_SAMPLES = 1000
STEPS = 2000
BATCH_SIZE = 64
LR = 0.01


# --------------------------------------------------------------------------- data

def make_sequence(rng, lo, hi):
    """One 'n op n op n ...' sequence with 2-5 operators; returns (tokens, signed sum)."""
    length = rng.randint(2, 5)
    curr = rng.randint(lo, hi)
    tokens, total = [curr], curr
    for _ in range(length):
        op = rng.choice(['+', '-'])
        num = rng.randint(lo, hi)
        tokens += [op, num]
        total += num if op == '+' else -num
    return tokens, total


def make_batch(rng, n, lo, hi):
    """n sequences, zero-padded on the right. Returns x [n, T, 6] and y [n]."""
    seqs = [make_sequence(rng, lo, hi) for _ in range(n)]
    T = max(len(toks) for toks, _ in seqs)
    x = torch.zeros(n, T, D)
    for i, (toks, _) in enumerate(seqs):
        x[i, :len(toks)] = torch.stack([get_tensor_embedding(t) for t in toks])
    y = torch.tensor([float(s) for _, s in seqs])
    return x, y


_EVAL_SETS = {}


def eval_set(which):
    """Fixed evaluation set shared by every configuration and seed ('id' or 'ood')."""
    if which not in _EVAL_SETS:
        lo, hi = ID_RANGE if which == 'id' else OOD_RANGE
        rng = random.Random(EVAL_SEED + (0 if which == 'id' else 1))
        _EVAL_SETS[which] = make_batch(rng, EVAL_SAMPLES, lo, hi)
    return _EVAL_SETS[which]


# -------------------------------------------------------------------------- model

class SigmoidGLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w_gate = nn.Linear(d, d)
        self.w_val = nn.Linear(d, d)

    def forward(self, x):
        return torch.sigmoid(self.w_gate(x)) * self.w_val(x)


class DenseBlock(nn.Module):
    """
    Single-head causal attention with ALiBi-style distance penalty, residual, sigmoid GLU,
    residual, sum pooling, linear readout (paper Section 2 / Config A).

    layernorm        Pre-LN before attention and before the GLU (Aln).
    gain             Multiply the attention output by this before the residual (Cg/Csb/Cc/Cb).
    gain_learnable   The gain is a trainable scalar (else a fixed buffer).
    oracle_attention Q/K/V fixed to the hand-coded routing of Appendix A and frozen (Bfa/Cb).
    frozen_logic     GLU and readout fixed to the hand-coded ALU of Appendix A and frozen (Bfl).
    gate_bias        Initial value for every gate bias (Cbi).
    """

    def __init__(self, layernorm=False, gain=None, gain_learnable=False,
                 oracle_attention=False, frozen_logic=False, gate_bias=None):
        super().__init__()
        # Creation order fixes how torch's RNG is consumed by initialisation; keep it.
        self.W_q = nn.Linear(D, D, bias=False)
        self.W_k = nn.Linear(D, D, bias=False)
        self.W_v = nn.Linear(D, D, bias=False)
        self.ffn = SigmoidGLU(D)
        self.W_out = nn.Linear(D, 1, bias=False)
        self.ln1 = nn.LayerNorm(D) if layernorm else None
        self.ln2 = nn.LayerNorm(D) if layernorm else None
        if gain is None:
            self.attn_gain = None
        elif gain_learnable:
            self.attn_gain = nn.Parameter(torch.tensor(float(gain)))
        else:
            self.register_buffer('attn_gain', torch.tensor(float(gain)))
        if gate_bias is not None:
            with torch.no_grad():
                self.ffn.w_gate.bias.fill_(gate_bias)
        if oracle_attention:
            set_oracle_attention(self)
        if frozen_logic:
            set_frozen_logic(self)
        self.last_attn = None

    def forward(self, x):
        T = x.shape[1]
        h = self.ln1(x) if self.ln1 is not None else x
        scores = self.W_q(h) @ self.W_k(h).transpose(-2, -1)
        idx = torch.arange(T, device=x.device)
        scores = scores - (idx.view(-1, 1) - idx.view(1, -1)).abs().float()
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn = torch.softmax(scores.masked_fill(causal, float('-inf')), dim=-1)
        self.last_attn = attn.detach()
        Z = attn @ self.W_v(h)
        if self.attn_gain is not None:
            Z = Z * self.attn_gain
        combined = x + Z
        ffn_in = self.ln2(combined) if self.ln2 is not None else combined
        out = combined + self.ffn(ffn_in)
        return self.W_out(out.sum(dim=1)).squeeze(-1)


def set_oracle_attention(model):
    """Appendix A routing: numbers (D1) query operators (D4); values copy IsAdd/IsSub."""
    with torch.no_grad():
        w_q = torch.zeros(D, D); w_q[4, 1] = 10.0
        w_k = torch.zeros(D, D); w_k[4, 4] = 10.0
        w_v = torch.zeros(D, D); w_v[2, 2] = 1.0; w_v[3, 3] = 1.0
        model.W_q.weight.copy_(w_q)
        model.W_k.weight.copy_(w_k)
        model.W_v.weight.copy_(w_v)
    for m in (model.W_q, model.W_k, model.W_v):
        m.weight.requires_grad = False


def set_frozen_logic(model, gain=100.0):
    """Appendix A ALU: gate opens on IsNum AND IsSub, copies D0 to D5; y = sum(D0 - 2*D5)."""
    with torch.no_grad():
        w_gate = torch.zeros(D, D); b_gate = torch.zeros(D)
        w_gate[5, 1] = gain; w_gate[5, 3] = gain; b_gate[5] = -1.5 * gain
        w_val = torch.zeros(D, D); w_val[5, 0] = 1.0
        w_out = torch.zeros(1, D); w_out[0, 0] = 1.0; w_out[0, 5] = -2.0
        model.ffn.w_gate.weight.copy_(w_gate); model.ffn.w_gate.bias.copy_(b_gate)
        model.ffn.w_val.weight.copy_(w_val); model.ffn.w_val.bias.zero_()
        model.W_out.weight.copy_(w_out)
    for p in model.ffn.parameters():
        p.requires_grad = False
    model.W_out.weight.requires_grad = False


def make_strat_lite():
    from strat_lite import STRATLite
    return STRATLite()


# ------------------------------------------------------------------ configurations

@dataclass(frozen=True)
class Spec:
    make_model: Callable[[], nn.Module]
    scale: float = 1.0          # divide D0 inputs and targets by this; multiply outputs back
    curriculum: bool = False    # widen the training range over time (curriculum_max)
    ortho_lambda: float = 0.0   # weight of sum(W_gate[:, D0]^2) added to the loss
    lr: float = LR


CONFIGS = {
    'A':          Spec(lambda: DenseBlock()),
    'As':         Spec(lambda: DenseBlock(), scale=5000.0),
    'Aln':        Spec(lambda: DenseBlock(layernorm=True)),
    'Acurr':      Spec(lambda: DenseBlock(), curriculum=True),
    'Bfl':        Spec(lambda: DenseBlock(frozen_logic=True)),
    'Bfa':        Spec(lambda: DenseBlock(oracle_attention=True)),
    'Bortho':     Spec(lambda: DenseBlock(), ortho_lambda=10.0),
    'Cg':         Spec(lambda: DenseBlock(gain=1.0, gain_learnable=True)),
    'Csb':        Spec(lambda: DenseBlock(gain=100.0)),
    'Cc':         Spec(lambda: DenseBlock(gain=50.0, gain_learnable=True), ortho_lambda=10.0),
    'Cb':         Spec(lambda: DenseBlock(gain=50.0, gain_learnable=True, oracle_attention=True)),
    'Cbi':        Spec(lambda: DenseBlock(gate_bias=1.0)),
    'STRAT-Lite': Spec(make_strat_lite),
}


def curriculum_max(step, steps):
    """Acurr schedule: top of the value range for this step (100, then 1000, then 5000)."""
    if step < steps // 4:
        return 100
    if step < steps // 2:
        return 1000
    return 5000


# ------------------------------------------------------------- train / evaluate

def scale_inputs(x, scale):
    if scale == 1.0:
        return x
    x = x.clone()
    x[..., 0] = x[..., 0] / scale
    return x


def build(name, seed):
    """Seed torch and return a freshly initialised model for configuration `name`."""
    torch.manual_seed(seed)
    return CONFIGS[name].make_model()


def train(name, seed, steps=STEPS, batch_size=BATCH_SIZE, on_step=None):
    """
    Train one run. Returns (model, MSE of the final training batch, data rng).
    on_step(step, model, x_in, mse), if given, is called after backward() and before
    gradient clipping, with the (scaled) input batch the model saw. It must not modify
    the model, its gradients, or any RNG.
    """
    spec = CONFIGS[name]
    model = build(name, seed)
    rng = random.Random(seed)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=spec.lr)
    model.train()
    for step in range(steps):
        hi = curriculum_max(step, steps) if spec.curriculum else ID_RANGE[1]
        x, y = make_batch(rng, batch_size, ID_RANGE[0], hi)
        x_in = scale_inputs(x, spec.scale)
        pred = model(x_in).reshape(-1)
        mse = F.mse_loss(pred, y / spec.scale)
        loss = mse
        if spec.ortho_lambda:
            loss = loss + spec.ortho_lambda * (model.ffn.w_gate.weight[:, 0] ** 2).sum()
        opt.zero_grad()
        loss.backward()
        if on_step is not None:
            on_step(step, model, x_in, mse.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model, mse.item(), rng


def predict(model, name, x):
    """Model output in raw units (undoes the As scaling)."""
    scale = CONFIGS[name].scale
    model.eval()
    with torch.no_grad():
        return model(scale_inputs(x, scale)).reshape(-1) * scale


def evaluate(model, name, x, y):
    """Returns (MAE, % of sequences with |error| < 0.01)."""
    err = (predict(model, name, x) - y).abs()
    return err.mean().item(), (err < 1e-2).float().mean().item() * 100
