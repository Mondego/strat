"""Game of Life tokens: one token per cell.

tokenize_onehot gives the type ids of the hand-written Type of the paper, tokenize_learned_type those of
the learned Type; tokenize picks one of them by Type mode. Both give the same Data.
"""

import torch

N_LEARNED_TYPES = 2
INPUT_ENCODING = "gol_alive_onehot_v1"


def onehot_table():
    """Hand-written Type of every tokenize_onehot id."""
    return torch.eye(2)


def tokenize_onehot(grid, d_data):
    """Type ids of the hand-written Type, and Data."""
    batch = grid.shape[0]
    alive = grid.reshape(batch, -1)
    type_ids = alive.long()
    data = torch.zeros(batch, alive.shape[1], d_data, device=grid.device)
    data[..., 0] = alive.float()
    return type_ids, data


def tokenize_learned_type(grid, d_data):
    """Type ids of the learned Type, and Data."""
    batch = grid.shape[0]
    alive = grid.reshape(batch, -1)
    type_ids = alive.long()
    data = torch.zeros(batch, alive.shape[1], d_data, device=grid.device)
    data[..., 0] = alive.float()
    return type_ids, data


def tokenize(grid: torch.Tensor, d_data: int, type_mode: str = "onehot"):
    """Return type ids [batch, cells] and Data [batch, cells, d_data]."""
    return {"onehot": tokenize_onehot, "learned": tokenize_learned_type}[type_mode](grid, d_data)
