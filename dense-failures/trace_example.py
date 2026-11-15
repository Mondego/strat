"""
Traces one expression through the hand-coded Geometric ALU of Appendix A (ablation.py's
DenseBlock with oracle attention and frozen logic) and prints the values used in the
Section 3 example table: input embeddings, attention weights, the operator flags each
token holds after attention, the gate on D5, D5 after the GLU, and the output.

Usage: python trace_example.py [--expr "3 + 5 - 7"]
Reads nothing; prints to stdout.

See trace_example.py.md.
"""
import argparse

import torch

import ablation


def embed(expr):
    rows = []
    for tok in expr.split():
        if tok == '+':
            rows.append([0, 0, 1, 0, 1, 0])
        elif tok == '-':
            rows.append([0, 0, 0, 1, 1, 0])
        else:
            rows.append([float(tok), 1, 0, 0, 0, 0])
    return torch.tensor([rows], dtype=torch.float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--expr', default='3 + 5 - 7')
    args = ap.parse_args()
    torch.set_printoptions(precision=3, sci_mode=False)
    m = ablation.DenseBlock(oracle_attention=True, frozen_logic=True).eval()
    x = embed(args.expr)
    with torch.no_grad():
        y = m(x)
        attn = m.last_attn[0]
        after_attn = x[0] + attn @ m.W_v(x)[0]
        gate = torch.sigmoid(m.ffn.w_gate(after_attn))[:, 5]
        out = after_attn + m.ffn(after_attn)
    print('input E(t):\n', x[0])
    print('attention (row = query token):\n', attn)
    print('after attention:\n', after_attn)
    print('gate on D5:', gate)
    print('after GLU:\n', out)
    print('sum D0 =', out[:, 0].sum().item(), ' sum D5 =', out[:, 5].sum().item(), ' y =', y.item())


if __name__ == '__main__':
    main()
