"""Tic-Tac-Toe tokens.

tokenize_onehot gives the tokens and type ids of the hand-written Type of the paper, tokenize_learned_type
those of the learned Type; tokenize picks one of them by Type mode.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

N_CELLS = 9
N_CLASSES = 2
N_LINES = 8
N_VALUES = 7
N_LEARNED_TYPES = 1 + N_CELLS


def build_type_template(d_ctrl=32):
    """Type of every token without the observed marks."""
    template = torch.zeros(1 + N_CELLS + N_LINES, d_ctrl)
    template[0, 0] = 1.0
    template[1:10, 1] = 1.0
    template[10:, 2] = 1.0
    for line in range(8):
        template[10 + line, 3 + min(line // 3, 2)] = 1.0
        template[10 + line, 6 + line] = 1.0
    for index in range(9):
        row, column = divmod(index, 3)
        template[1 + index, 6 + row] = 1.0
        template[1 + index, 9 + column] = 1.0
        template[1 + index, 12] = float(row == column)
        template[1 + index, 13] = float(row + column == 2)
    return template


def onehot_table(d_ctrl=32):
    """Hand-written Type of every tokenize_onehot id."""
    template = build_type_template(d_ctrl)
    cells = template[1:10].repeat_interleave(N_VALUES, dim=0)
    cells[:, -N_VALUES:] = torch.eye(N_VALUES).repeat(N_CELLS, 1)
    return torch.cat([template[:1], cells, template[10:]])


def tokenize_onehot(values, d_data, n_heads):
    """Tokens and type ids of the hand-written Type, and Data."""
    block = d_data // n_heads
    batch = values.shape[0]
    cells = 1 + N_VALUES * torch.arange(N_CELLS, device=values.device) + values
    lines = 1 + N_VALUES * N_CELLS + torch.arange(N_LINES, device=values.device)
    type_ids = torch.cat([values.new_zeros(batch, 1), cells, lines.expand(batch, -1)], dim=1)
    per_head = torch.zeros(batch, 1 + N_CELLS + N_LINES, block, device=values.device)
    per_head[:, 1:1 + N_CELLS, :7] = F.one_hot(values, 7).float()
    per_head[:, 1:1 + N_CELLS, 7] = 1.0
    return type_ids, per_head.repeat(1, 1, n_heads)


def tokenize_learned_type(values, d_data, n_heads):
    """Tokens and type ids of the learned Type, and Data."""
    block = d_data // n_heads
    batch = values.shape[0]
    type_ids = torch.arange(1 + N_CELLS, device=values.device).expand(batch, -1)
    per_head = torch.zeros(batch, 1 + N_CELLS, block, device=values.device)
    per_head[:, 1:, :7] = F.one_hot(values, 7).float()
    per_head[:, 1:, 7] = 1.0
    return type_ids, per_head.repeat(1, 1, n_heads)


def tokenize(values: torch.Tensor, d_data: int, n_heads: int, type_mode: str = "onehot"):
    """Return type ids [batch, tokens] and Data [batch, tokens, d_data]."""
    return {"onehot": tokenize_onehot, "learned": tokenize_learned_type}[type_mode](values, d_data, n_heads)
