"""Shared Boolean inputs, formula splits, sampling, training, and evaluation.

Both models receive the same token sequence from tokenizer.py.
"""
import argparse
import json
import os
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from .strat import STRAT
from .tokenizer import tokenize
from .vanilla import Vanilla

DATASET = 'satbench'
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', DATASET)
TRAIN_PATH = os.path.join(DATA_DIR, 'id', 'satbench_train.jsonl')
TEST_PATH = os.path.join(DATA_DIR, 'id', 'satbench_test.jsonl')
EPOCHS = 2000
LEARNING_RATE = 0.01
BATCH_SIZE = 16


def eval_clauses(clauses, assignment):
    """Truth value of every clause under the assignment."""
    out = []
    for cl in clauses:
        sat = False
        for lit in cl:
            v = abs(lit)
            if (lit > 0 and assignment[v]) or (lit < 0 and not assignment[v]):
                sat = True
                break
        out.append(sat)
    return out


def find_sat(clauses, n_vars):
    """A satisfying assignment from pycosat."""
    import pycosat

    sol = pycosat.solve(clauses)
    a = [False] * (n_vars + 1)
    for lit in sol:
        if abs(lit) <= n_vars:
            a[abs(lit)] = lit > 0
    return a


def perturb(assignment, k, rng):
    """Flip exactly k random bits and return a new list."""
    n = len(assignment) - 1
    out = assignment[:]
    for v in rng.sample(range(1, n + 1), k):
        out[v] = not out[v]
    return out


def random_assignment(n_vars, rng):
    return [False] + [rng.random() < 0.5 for _ in range(n_vars)]


def load_train():
    with open(TRAIN_PATH) as f:
        return [json.loads(l) for l in f]


def load_test():
    with open(TEST_PATH) as f:
        return [json.loads(l) for l in f]


def experiment_split(n_train=10):
    """Return disjoint formula pools: the first formulas of the training file and the test file."""
    return load_train()[:n_train], load_test()


def _full_test(instances, seed=0):
    """One labeled sample per formula: a satisfying assignment if the formula is satisfiable, else a random one."""
    rng = random.Random(seed)
    out = []
    for inst in instances:
        nv, cls = inst['num_vars'], inst['clauses']
        if inst['satisfiable']:
            a = find_sat(cls, nv)
        else:
            a = random_assignment(nv, rng)
        out.append((cls, a, nv, int(all(eval_clauses(cls, a)))))
    return out


class SampleBank:
    """Training samples from the satisfiable formulas: satisfying assignments and perturbed, non-satisfying ones."""

    def __init__(self, instances):
        self.entries = [dict(n_vars=inst['num_vars'], clauses=inst['clauses'],
                             sat_asgn=find_sat(inst['clauses'], inst['num_vars'])) for inst in instances]

    def _pos(self, rng):
        e = rng.choice(self.entries)
        a = e['sat_asgn'][:]
        if rng.random() < 0.3:
            for _ in range(5):
                cand = perturb(a, 1, rng)
                if all(eval_clauses(e['clauses'], cand)):
                    a = cand
                    break
        return (e['clauses'], a, e['n_vars'], 1)

    def _neg(self, rng):
        e = rng.choice(self.entries)
        a = None
        for k in (1, 2, 3):
            for _ in range(10):
                cand = perturb(e['sat_asgn'], k, rng)
                if not all(eval_clauses(e['clauses'], cand)):
                    a = cand
                    break
            if a is not None:
                break
        return (e['clauses'], a if a is not None else perturb(e['sat_asgn'], 5, rng),
                e['n_vars'], 0)

    def sample_raw(self, n_pos, n_neg, rng):
        return [self._pos(rng) for _ in range(n_pos)] + [self._neg(rng) for _ in range(n_neg)]


def stack(samples):
    """Right-pad both models' variable-length sequences with zero-valued PAD fields."""
    out = {}
    for k in samples[0]:
        if k == 'label':
            continue
        L = max(s[k].shape[0] for s in samples)
        out[k] = torch.stack([F.pad(s[k], (0, L - s[k].shape[0])) for s in samples])
    out['label'] = torch.stack([s['label'] for s in samples])
    if 'mask' in out:
        L = int(out['mask'].sum(dim=1).max())
        for k in out:
            if k != 'label':
                out[k] = out[k][:, :L]
    return out


def collate(raw, type_mode):
    """Tokenize every (clauses, assignment, n_vars, label) sample and pad the batch."""
    items = []
    for clauses, assignment, _, label in raw:
        item = tokenize(clauses, assignment, type_mode)
        item['label'] = torch.tensor(float(label))
        items.append(item)
    return stack(items)


def _accuracy(model, batch, dev, chunk=256):
    model.eval()
    with torch.no_grad():
        preds = []
        keys = [k for k in batch if k != 'label']
        for i in range(0, batch['label'].shape[0], chunk):
            inp = {k: batch[k][i:i + chunk].to(dev) for k in keys}
            preds.append((model(**inp) > 0).float().cpu())
        pred, lab = torch.cat(preds), batch['label']
        return float((pred == lab).float().mean())


def run(args):
    """Train STRAT or the Transformer, save it, and score it on the test pool."""
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = (STRAT if args.method == 'strat' else Vanilla)(type_mode=args.type_mode).to(dev)
    rng = random.Random(args.seed)
    train_instances, test_instances = experiment_split()
    bank = SampleBank(train_instances)
    test_eval = collate(_full_test(test_instances, seed=args.seed + 1000), args.type_mode)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce = torch.nn.BCEWithLogitsLoss()
    for ep in range(args.epochs):
        model.train()
        batch = collate(bank.sample_raw(BATCH_SIZE, BATCH_SIZE, rng), args.type_mode)
        inp = {k: v.to(dev) for k, v in batch.items() if k != 'label'}
        loss = bce(model(**inp), batch['label'].to(dev))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ep % 500 == 0 or ep == args.epochs - 1:
            print('%s/%s seed=%d epoch=%d loss=%.4f' % (DATASET, args.method, args.seed, ep, loss.item()), flush=True)

    accuracy = _accuracy(model, test_eval, dev, chunk=64)
    path = Path(__file__).parent / 'results' / args.type_mode / f'{args.method}_seed{args.seed}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'meta': {'type_mode': args.type_mode}, 'method': args.method, 'seed': args.seed,
                'epochs': args.epochs, 'accuracy': accuracy}, path)
    print('%s/%s seed=%d accuracy=%.4f' % (DATASET, args.method, args.seed, accuracy), flush=True)


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
