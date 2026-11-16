"""Sudoku tokens: one token per cell.

tokenize_onehot gives the type ids of the hand-written Type of the paper, tokenize_learned_type those of
the learned Type; tokenize picks one of them by Type mode. Both give the same Data.
"""
import torch
import torch.nn.functional as F

N_DIGITS = 9
N_CELLS = 81
N_LEARNED_TYPES = N_CELLS


def onehot_table():
    """Hand-written Type of every tokenize_onehot id."""
    cell = torch.arange(N_CELLS).repeat_interleave(N_DIGITS + 1)
    digit = torch.arange(N_DIGITS + 1).repeat(N_CELLS)
    r, c = cell // 9, cell % 9
    b = (r // 3) * 3 + (c // 3)
    given = F.one_hot(digit, N_DIGITS + 1)[:, 1:]
    return torch.cat([F.one_hot(r, 9), F.one_hot(c, 9), F.one_hot(b, 9), (digit > 0).unsqueeze(-1), given], -1).float()


def tokenize_onehot(givens, d_data):
    """Type ids of the hand-written Type, and Data."""
    cell = torch.arange(N_CELLS, device=givens.device)
    digit = (givens.argmax(-1) + 1) * (givens.sum(-1) > 0)
    type_ids = (N_DIGITS + 1) * cell + digit
    D = givens.new_zeros(givens.shape[0], N_CELLS, d_data)
    D[..., :N_DIGITS] = givens
    return type_ids, D


def tokenize_learned_type(givens, d_data):
    """Type ids of the learned Type, and Data."""
    type_ids = torch.arange(N_CELLS, device=givens.device).expand(givens.shape[0], -1)
    D = givens.new_zeros(givens.shape[0], N_CELLS, d_data)
    D[..., :N_DIGITS] = givens
    return type_ids, D


def tokenize(givens, d_data, type_mode='onehot'):
    """givens [B, cells, digits] one-hot -> type ids [B, cells], Data [B, cells, d_data]."""
    return {'onehot': tokenize_onehot, 'learned': tokenize_learned_type}[type_mode](givens, d_data)
