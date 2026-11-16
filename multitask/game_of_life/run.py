"""Data, training and evaluation of STRAT and the Transformer on Game of Life."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .strat import STRAT, StratCfg
from .vanilla import Vanilla, VanillaCfg

DATA_FILE = Path(__file__).parent / "../data/game_of_life/id/gol6x6.npz"
EPOCHS = 6000
LEARNING_RATE = 1e-3


def load_data():
    """Training transitions and ID test transitions."""
    raw = np.load(DATA_FILE)
    x = torch.tensor(raw["X"]).float()
    y = torch.tensor(raw["Y"]).float()
    return x[:10], y[:10], x[10:], y[10:]


def augment(x, y, generator):
    """One shared torus shift, rotation, and reflection for the full batch."""
    height, width = x.shape[1:]
    dr = int(torch.randint(0, height, (1,), generator=generator).item())
    dc = int(torch.randint(0, width, (1,), generator=generator).item())
    turns = int(torch.randint(0, 4, (1,), generator=generator).item())
    flip_rows = bool(torch.rand((), generator=generator).item() < 0.5)
    flip_cols = bool(torch.rand((), generator=generator).item() < 0.5)

    x = torch.roll(x, (dr, dc), (1, 2))
    y = torch.roll(y, (dr, dc), (1, 2))
    x = torch.rot90(x, turns, (1, 2))
    y = torch.rot90(y, turns, (1, 2))
    if flip_rows:
        x, y = x.flip(1), y.flip(1)
    if flip_cols:
        x, y = x.flip(2), y.flip(2)
    return x, y


@torch.no_grad()
def evaluate(model, x, y):
    """Fraction of boards whose cells are all predicted correctly."""
    model.eval()
    logits, _ = model(x)
    prediction = logits.gt(0).reshape_as(y)
    correct = prediction.eq(y.bool())
    return correct.flatten(1).all(1).float().mean().item()


def run(args):
    """Train STRAT or the Transformer on the training boards, save it, and score the test boards."""
    torch.set_num_threads(1)
    device = torch.device(args.device)
    x_train, y_train, x_test, y_test = (tensor.to(device) for tensor in load_data())
    torch.manual_seed(args.seed)
    cls, cfg = (STRAT, StratCfg) if args.method == "strat" else (Vanilla, VanillaCfg)
    model = cls(cfg(type_mode=args.type_mode)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs):
        model.train()
        batch_x, batch_y = augment(x_train, y_train, generator)
        logits, _ = model(batch_x)
        loss = F.binary_cross_entropy_with_logits(logits, batch_y.reshape(len(batch_y), -1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch % 500 == 0 or epoch == args.epochs - 1:
            print(f"game_of_life/{args.method} seed={args.seed} epoch={epoch} loss={loss.item():.7f}", flush=True)

    train_board, test_board = evaluate(model, x_train, y_train), evaluate(model, x_test, y_test)
    path = Path(__file__).parent / "results" / args.type_mode / f"{args.method}_seed{args.seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "architecture": asdict(model.cfg),
        "method": args.method,
        "seed": args.seed,
        "epochs": args.epochs,
        "train_board": train_board,
        "test_board": test_board,
    }, path)
    print(f"game_of_life/{args.method} seed={args.seed} test_board={test_board:.4f}", flush=True)


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
