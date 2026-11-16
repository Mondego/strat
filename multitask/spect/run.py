"""Full-batch training for a fixed number of epochs, then ID and OOD evaluation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .strat import STRAT, StratCfg
from .vanilla import Vanilla, VanillaCfg

TASK = 'spect'
DATA_DIR = Path(__file__).parent / '../data/spect'
EPOCHS = 2000
LEARNING_RATE = 0.001


def load_splits():
    """Splits train and id come from the ID file, split ood from the OOD file."""
    splits = {}
    for part in ('id', 'ood'):
        splits.update(json.loads((DATA_DIR / part / 'spect.json').read_text())['splits'])
    return splits


@torch.inference_mode()
def predict(model, values, device, batch_size=256):
    model.eval()
    values = torch.as_tensor(values, dtype=torch.long)
    return torch.cat([model(x.to(device))[0].cpu() for x in values.split(batch_size)])


def score(logits, split):
    """Accuracy and cross-entropy, weighted when the split carries weights."""
    labels = torch.tensor(split['labels'], dtype=torch.long)
    weights = torch.tensor(split.get('weights', [1.0] * len(labels)), dtype=torch.float64)
    correct = logits.argmax(-1).eq(labels).double()
    loss = F.cross_entropy(logits, labels, reduction='none').double()
    return {'accuracy': float((correct * weights).sum() / weights.sum()),
            'cross_entropy': float((loss * weights).sum() / weights.sum())}


def run(args):
    """Train STRAT or the Transformer on the train split, save it, and score every split."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    splits = load_splits()
    torch.manual_seed(args.seed)
    cls, cfg = (STRAT, StratCfg) if args.method == 'strat' else (Vanilla, VanillaCfg)
    model = cls(cfg(type_mode=args.type_mode)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=.01)
    values = torch.tensor(splits['train']['values'], dtype=torch.long, device=device)
    labels = torch.tensor(splits['train']['labels'], dtype=torch.long, device=device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits, _ = model(values)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch % 500 == 0 or epoch == args.epochs:
            print(f'{TASK}/{args.method} seed={args.seed} epoch={epoch} loss={float(loss.detach()):.7f}', flush=True)
    metrics = {name: score(predict(model, split['values'], device), split) for name, split in splits.items()}
    path = Path(__file__).parent / 'results' / args.type_mode / f'{args.method}_seed{args.seed}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'payload': {'method': args.method, 'seed': args.seed, 'epochs': args.epochs,
                            'model_config': asdict(model.cfg), 'metrics': metrics}}, path)
    print(f"{TASK}/{args.method} seed={args.seed} id={metrics['id']['accuracy']:.4f} ood={metrics['ood']['accuracy']:.4f}", flush=True)


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
