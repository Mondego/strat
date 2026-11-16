"""Unityped Transformer baseline for Game of Life."""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import N_LEARNED_TYPES, onehot_table, tokenize
from .topology import D4RelativeBias
from .type_embedding import TypeEmbedding


@dataclass(frozen=True)
class VanillaCfg:
    height: int = 6
    width: int = 6
    n_heads: int = 4
    d_ctrl: int = 32
    d_data: int = 32
    d_model: int = 64
    hidden: int = 68
    type_mode: str = "onehot"
    type_width: int = 5

    @property
    def d_head(self):
        return self.d_model // self.n_heads


class Vanilla(nn.Module):
    """Standard attention and FFN over the whole stream."""

    def __init__(self, cfg: VanillaCfg | None = None):
        super().__init__()
        self.cfg = cfg or VanillaCfg()
        cfg = self.cfg
        self.n_tokens = cfg.height * cfg.width

        raw_dim = cfg.d_ctrl + cfg.d_data
        self.input_proj = nn.Identity() if cfg.d_model == raw_dim else nn.Linear(raw_dim, cfg.d_model)
        self.norm_attention = nn.LayerNorm(cfg.d_model)
        self.W_q = nn.Linear(cfg.d_model, cfg.d_model)
        self.W_k = nn.Linear(cfg.d_model, cfg.d_model)
        self.W_v = nn.Linear(cfg.d_model, cfg.d_model)
        self.W_o = nn.Linear(cfg.d_model, cfg.d_model)
        self.relative_bias = D4RelativeBias(cfg.n_heads, cfg.height, cfg.width)
        self.norm_ffn = nn.LayerNorm(cfg.d_model)
        self.ffn_1 = nn.Linear(cfg.d_model, cfg.hidden)
        self.ffn_mid = nn.Linear(cfg.hidden, cfg.hidden)
        self.ffn_2 = nn.Linear(cfg.hidden, cfg.d_model)
        self.norm_out = nn.LayerNorm(cfg.d_model)
        self.readout = nn.Linear(cfg.d_model, 1)

        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.zeros_(self.W_q.bias)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.zeros_(self.W_k.bias)
        nn.init.eye_(self.W_v.weight)
        nn.init.zeros_(self.W_v.bias)
        nn.init.zeros_(self.W_o.weight)
        nn.init.zeros_(self.W_o.bias)
        nn.init.zeros_(self.ffn_2.weight)
        nn.init.zeros_(self.ffn_2.bias)

        self.type_embedding = TypeEmbedding(cfg.type_mode, onehot_table(), N_LEARNED_TYPES, cfg.type_width, cfg.d_ctrl)

    def forward(self, grid):
        cfg = self.cfg
        # The same [Type | Data] fields as STRAT.
        type_ids, data = tokenize(grid, cfg.d_data, cfg.type_mode)
        raw_stream = torch.cat([self.type_embedding(type_ids), data], dim=-1)
        stream = self.input_proj(raw_stream)
        batch = grid.shape[0]
        shape = (batch, self.n_tokens, cfg.n_heads, cfg.d_head)

        normed = self.norm_attention(stream)
        query = self.W_q(normed).view(*shape).transpose(1, 2)
        key = self.W_k(normed).view(*shape).transpose(1, 2)
        value = self.W_v(normed).view(*shape).transpose(1, 2)
        scores = query @ key.transpose(-2, -1) / math.sqrt(cfg.d_head)
        attention = torch.softmax(scores + self.relative_bias().unsqueeze(0), dim=-1)
        routed = (attention @ value).transpose(1, 2).reshape(batch, self.n_tokens, cfg.d_model)
        stream = stream + self.W_o(routed)

        hidden = F.gelu(self.ffn_1(self.norm_ffn(stream)))
        hidden = F.gelu(self.ffn_mid(hidden))
        stream = stream + self.ffn_2(hidden)
        logits = self.readout(self.norm_out(stream)).squeeze(-1)

        return logits, {}
