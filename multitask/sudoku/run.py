"""Data, augmentation, metric and training loop shared by STRAT and the Transformer on Sudoku."""
import argparse
from dataclasses import asdict
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from .strat import StratCfg, SudokuStrat
from .vanilla import SudokuVanilla, VanillaCfg

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'sudoku', 'id')
N_DIGITS = 9
N_CELLS = 81
EPOCHS = 6000
BATCH_SIZE = 128
LEARNING_RATE = 1e-3


def load_pt(name):
    """Givens as one-hot digits (all zero for an empty cell) and the solution of every puzzle."""
    X = torch.load(os.path.join(DATA_DIR, name + '_features.pt'), weights_only=False)
    Y = torch.load(os.path.join(DATA_DIR, name + '_labels.pt'), weights_only=False)
    n = X.shape[0]
    return X.reshape(n, N_CELLS, N_DIGITS).float(), Y.reshape(n, N_CELLS, N_DIGITS).argmax(-1).long()


def load_splits():
    """Training puzzles and ID test puzzles."""
    gtr, ttr = load_pt('train')
    gte, tte = load_pt('test')
    return gtr[:10], ttr[:10], torch.cat([gtr[10:], gte]), torch.cat([ttr[10:], tte])


def _row_perm(g):
    """Permute the bands and the rows inside each band, which keeps the boxes."""
    out = []
    for b in torch.randperm(3, generator=g).tolist():
        for w in torch.randperm(3, generator=g).tolist():
            out.append(b * 3 + w)
    return torch.tensor(out, dtype=torch.long)


def augment_batch(gb, tb, g):
    """Rule-preserving augmentation: digit relabeling, row and column band permutations, transpose."""
    out_g = torch.zeros_like(gb)
    out_t = torch.zeros_like(tb)
    gd = gb.argmax(-1) + 1
    gd[gb.sum(-1) == 0] = 0                          # givens as digits, zero for an empty cell
    sd = tb + 1                                      # solution as digits
    for b in range(gb.shape[0]):
        dp = torch.randperm(9, generator=g) + 1
        gg, ss = gd[b].view(9, 9).clone(), sd[b].view(9, 9).clone()
        m = gg > 0
        gg[m] = dp[gg[m] - 1]
        ss = dp[ss - 1]
        rp, cp = _row_perm(g), _row_perm(g)
        gg, ss = gg[rp][:, cp], ss[rp][:, cp]
        if torch.rand((), generator=g) < 0.5:
            gg, ss = gg.t().contiguous(), ss.t().contiguous()
        gg, ss = gg.reshape(N_CELLS), ss.reshape(N_CELLS)
        giv = gg > 0
        out_g[b][giv] = F.one_hot((gg[giv] - 1).long(), N_DIGITS).float()
        out_t[b] = (ss - 1).long()
    return out_g, out_t


@torch.no_grad()
def evaluate(model, givens, sol, dev, bs=128):
    """Whole-board accuracy: the givens are kept and every cell must be correct."""
    model.eval()
    n = givens.shape[0]
    board = 0
    for i in range(0, n, bs):
        g, t = givens[i:i + bs].to(dev), sol[i:i + bs].to(dev)
        logits, _ = model(g)
        is_given = g.sum(-1) > 0
        filled = torch.where(is_given, g.argmax(-1), logits.argmax(-1))
        board += int((filled == t).all(1).sum())
    return board / n


def run(args):
    """Train STRAT or the Transformer, save it, and score the test puzzles; the loss covers the empty cells."""
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    cls, cfg = (SudokuStrat, StratCfg) if args.method == 'strat' else (SudokuVanilla, VanillaCfg)
    model = cls(cfg(type_mode=args.type_mode)).to(dev)
    gtr, ttr, gte, tte = load_splits()
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    g = torch.Generator().manual_seed(args.seed)
    n = gtr.shape[0]

    for ep in range(args.epochs):
        model.train()
        tot, seen = 0., 0
        for i in range(0, n, BATCH_SIZE):
            idx = torch.randperm(n, generator=g)[i:i + BATCH_SIZE]
            gb, tb = augment_batch(gtr[idx], ttr[idx], g)
            gb, tb = gb.to(dev), tb.to(dev)
            opt.zero_grad()
            logits, _ = model(gb)
            target = tb.clone()
            target[gb.sum(-1) > 0] = -100                         # only the empty cells are supervised
            loss = F.cross_entropy(logits.reshape(-1, N_DIGITS), target.reshape(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx); seen += len(idx)
        if ep % 500 == 0 or ep == args.epochs - 1:
            print('sudoku/%s seed=%d epoch=%d loss=%.3f' % (args.method, args.seed, ep, tot / seen), flush=True)

    board = evaluate(model, gte, tte, dev, BATCH_SIZE)
    path = Path(__file__).parent / 'results' / args.type_mode / f'{args.method}_seed{args.seed}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'cfg': asdict(model.cfg), 'method': args.method, 'seed': args.seed,
                'epochs': args.epochs, 'board': board}, path)
    print('sudoku/%s seed=%d board=%.4f' % (args.method, args.seed, board), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=('strat', 'transformer'), default='strat')
    parser.add_argument('--type-mode', choices=('onehot', 'learned'), default='onehot')
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--device', default='cpu')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
