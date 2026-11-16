from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import N_DIGITS, N_LEARNED_TYPES, onehot_table, tokenize
from .type_embedding import TypeEmbedding


@dataclass
class VanillaCfg:
    n_layers: int = 16
    n_heads: int = 4
    d_data: int = 18            # same Data width as STRAT
    d_ctrl: int = 64            # same Type width as STRAT
    d_model: int = 84           # Data and Type widths, zero-padded so that the heads divide it
    val_hidden: int = 128       # FFN hidden width, same as the STRAT ALU
    type_mode: str = 'onehot'
    type_width: int = 37        # width of the learned Type table


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.linear_2(self.act(self.linear_1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_hidden):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardBlock(d_model, d_hidden)

        # Every sublayer starts as the identity, as in STRAT.
        nn.init.zeros_(self.W_o.weight)
        nn.init.zeros_(self.ffn.linear_2.weight)
        nn.init.zeros_(self.ffn.linear_2.bias)

    def forward(self, x):
        B, N, _ = x.shape

        # Attention and the FFN read and update the whole stream.
        h = self.ln1(x)
        shape = (B, N, self.n_heads, self.d_head)
        Q = self.W_q(h).view(*shape).transpose(1, 2)
        K = self.W_k(h).view(*shape).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_weights = torch.softmax(scores, dim=-1)

        V = self.W_v(h).view(*shape).transpose(1, 2)
        Z = torch.matmul(attn_weights, V).transpose(1, 2).reshape(B, N, self.d_model)

        x = x + self.W_o(Z)
        x = x + self.ffn(self.ln2(x))
        return x


class SudokuVanilla(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.val_hidden)
            for _ in range(cfg.n_layers)
        ])
        self.ln_out = nn.LayerNorm(cfg.d_model)
        self.W_out = nn.Linear(cfg.d_model, N_DIGITS)
        self.type_embedding = TypeEmbedding(cfg.type_mode, onehot_table(), N_LEARNED_TYPES, cfg.type_width, cfg.d_ctrl)

    def forward(self, givens):
        """givens [B, cells, digits] -> (logits [B, cells, digits], unused zero), the interface of STRAT."""
        cfg = self.cfg
        type_ids, D = tokenize(givens, cfg.d_data, cfg.type_mode)
        T = self.type_embedding(type_ids)
        x = torch.cat([D, T], dim=-1)                                  # the same Data and Type channels as STRAT
        x = F.pad(x, (0, cfg.d_model - cfg.d_data - cfg.d_ctrl))

        for block in self.blocks:
            x = block(x)

        res = self.W_out(self.ln_out(x))
        return res, givens.new_zeros(())
