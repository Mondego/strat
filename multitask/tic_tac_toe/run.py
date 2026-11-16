"""Data, augmentation, training and evaluation of STRAT and the Transformer on Tic-Tac-Toe."""
import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .strat import STRAT, StratCfg
from .vanilla import Vanilla, VanillaCfg

DATA_FILE = Path(__file__).parent / "../data/tic_tac_toe/id/tic-tac-toe.data"
SELECTION_SEED = 20260801
TRAIN_SIZE = 10
EPOCHS = 2000
LEARNING_RATE = 1e-3


@dataclass
class LoadedTask:
    train_values: torch.Tensor
    train_labels: torch.Tensor
    test_values: torch.Tensor
    test_labels: torch.Tensor
    value_names: list
    label_names: list


def read_rows():
    """The cells and the label of every board, in file order."""
    with DATA_FILE.open(newline="", encoding="utf-8") as handle:
        return [(fields[:-1], fields[-1]) for fields in csv.reader(handle) if fields]


def select_covering(rows):
    """Training rows that cover both labels and as many (cell, mark) pairs as possible; ties follow a fixed permutation."""
    labels = sorted({label for _, label in rows})
    counts = Counter(label for _, label in rows)
    order = np.random.default_rng(SELECTION_SEED).permutation(len(rows)).tolist()
    rank = {index: position for position, index in enumerate(order)}
    selected = []
    covered = set()

    def add_best(candidates):
        candidates = [index for index in candidates if index not in selected]
        best = max(
            candidates,
            key=lambda index: (
                sum((column, value) not in covered for column, value in enumerate(rows[index][0])),
                -rank[index],
            ),
        )
        selected.append(best)
        covered.update((column, value) for column, value in enumerate(rows[best][0]))

    for label in labels:
        add_best([index for index, (_, row_label) in enumerate(rows) if row_label == label])
    while len(selected) < TRAIN_SIZE:
        selected_counts = Counter(rows[index][1] for index in selected)
        add_best([
            index for index, (_, label) in enumerate(rows)
            if index not in selected and counts[label] - selected_counts[label] > 1
        ])
    return selected


def d4_rows(features):
    grid = np.array(features, dtype=object).reshape(3, 3)
    outputs = []
    for turns in range(4):
        rotated = np.rot90(grid, turns)
        outputs.append(rotated.reshape(-1).tolist())
        outputs.append(np.fliplr(rotated).reshape(-1).tolist())
    return outputs


def augment(rows):
    """All symmetries of the square for every row, without duplicates, in first-occurrence order."""
    augmented = [(tuple(variant), label) for features, label in rows for variant in d4_rows(list(features))]
    return list(dict.fromkeys(augmented))


def load_task():
    pool = [(tuple(features), label) for features, label in read_rows()]
    selected = select_covering(pool)
    train_rows = augment([pool[index] for index in selected])
    train_inputs = {features for features, _ in train_rows}
    test_rows = [row for index, row in enumerate(pool) if index not in selected and row[0] not in train_inputs]

    value_names = [sorted({features[column] for features, _ in pool + train_rows}) for column in range(9)]
    value_maps = [{value: index for index, value in enumerate(names)} for names in value_names]
    label_names = sorted({label for _, label in pool})
    label_map = {label: index for index, label in enumerate(label_names)}

    def encode(rows):
        values = torch.tensor([[value_maps[column][value] for column, value in enumerate(features)]
                               for features, _ in rows], dtype=torch.long)
        return values, torch.tensor([label_map[label] for _, label in rows], dtype=torch.long)

    return LoadedTask(*encode(train_rows), *encode(test_rows), value_names, label_names)


@torch.no_grad()
def evaluate(model, values, labels, device, batch_size=512):
    model.eval()
    correct = 0
    for start in range(0, len(values), batch_size):
        batch_values = values[start:start + batch_size].to(device)
        batch_labels = labels[start:start + batch_size].to(device)
        logits, _ = model(batch_values)
        correct += int(logits.argmax(dim=-1).eq(batch_labels).sum())
    return correct / len(values)


def run(args):
    """Train STRAT or the Transformer on the augmented boards, save it, and score the held-out boards."""
    torch.set_num_threads(1)
    device = torch.device(args.device)
    task = load_task()
    torch.manual_seed(args.seed)
    cls, cfg = (STRAT, StratCfg) if args.method == "strat" else (Vanilla, VanillaCfg)
    model = cls(cfg(type_mode=args.type_mode)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    train_values = task.train_values.to(device)
    train_labels = task.train_labels.to(device)
    for epoch in range(args.epochs):
        model.train()
        logits, _ = model(train_values)
        loss = F.cross_entropy(logits, train_labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch % 500 == 0 or epoch == args.epochs - 1:
            print(f"tic_tac_toe/{args.method} seed={args.seed} epoch={epoch} loss={loss.item():.7f}", flush=True)

    payload = {
        "method": args.method,
        "seed": args.seed,
        "epochs": args.epochs,
        "architecture": asdict(model.cfg),
        "train_accuracy": evaluate(model, task.train_values, task.train_labels, device),
        "test_accuracy": evaluate(model, task.test_values, task.test_labels, device),
    }
    path = Path(__file__).parent / "results" / args.type_mode / f"{args.method}_seed{args.seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "payload": payload}, path)
    print(f"tic_tac_toe/{args.method} seed={args.seed} test_accuracy={payload['test_accuracy']:.4f}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("strat", "transformer"), default="strat")
    parser.add_argument("--type-mode", choices=("onehot", "learned"), default="onehot")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
