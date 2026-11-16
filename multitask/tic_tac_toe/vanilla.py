"""Unityped Transformer control for Tic-Tac-Toe."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import N_CLASSES, N_LEARNED_TYPES, onehot_table, tokenize
from .type_embedding import TypeEmbedding


@dataclass(frozen=True)
class VanillaCfg:
    n_layers: int = 2
    n_heads: int = 4
    d_ctrl: int = 32
    d_data: int = 32
    d_model: int = 64
    hidden: int = 64
    qk_scale: float = 1.0
    type_mode: str = "onehot"
    type_width: int = 5

    @property
    def d_head(self):
        return self.d_model // self.n_heads


class TransformerLayer(nn.Module):
    def __init__(self, cfg: VanillaCfg):
        super().__init__()
        self.cfg = cfg
        self.norm_attention = nn.LayerNorm(cfg.d_model)
        self.W_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.norm_ffn = nn.LayerNorm(cfg.d_model)
        self.ffn_1 = nn.Linear(cfg.d_model, cfg.hidden)
        self.ffn_mid = nn.Linear(cfg.hidden, cfg.hidden)
        self.ffn_2 = nn.Linear(cfg.hidden, cfg.d_model)

        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        with torch.no_grad():
            self.W_q.weight.mul_(cfg.qk_scale)
            self.W_k.weight.mul_(cfg.qk_scale)
        nn.init.eye_(self.W_v.weight)
        nn.init.zeros_(self.W_o.weight)
        nn.init.zeros_(self.ffn_2.weight)
        nn.init.zeros_(self.ffn_2.bias)

    def forward(self, stream):
        cfg = self.cfg
        batch, n_tokens, _ = stream.shape
        shape = (batch, n_tokens, cfg.n_heads, cfg.d_head)
        normed = self.norm_attention(stream)
        query = self.W_q(normed).view(*shape).transpose(1, 2)
        key = self.W_k(normed).view(*shape).transpose(1, 2)
        value = self.W_v(normed).view(*shape).transpose(1, 2)
        scores = query @ key.transpose(-2, -1) / math.sqrt(cfg.d_head)
        attention = torch.softmax(scores, dim=-1)
        routed = (attention @ value).transpose(1, 2).reshape(
            batch, n_tokens, cfg.d_model
        )
        stream = stream + self.W_o(routed)
        hidden = F.gelu(self.ffn_1(self.norm_ffn(stream)))
        hidden = F.gelu(self.ffn_mid(hidden))
        return stream + self.ffn_2(hidden)


class Vanilla(nn.Module):
    def __init__(self, cfg: VanillaCfg):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([TransformerLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm_out = nn.LayerNorm(cfg.d_model)
        self.readout = nn.Linear(cfg.d_model, N_CLASSES)
        self.type_embedding = TypeEmbedding(cfg.type_mode, onehot_table(cfg.d_ctrl), N_LEARNED_TYPES,
                                            cfg.type_width, cfg.d_ctrl)

    def forward(self, values):
        type_ids, data = tokenize(values, self.cfg.d_data, self.cfg.n_heads, self.cfg.type_mode)
        type_state = self.type_embedding(type_ids)
        stream = torch.cat([data, type_state], dim=-1)
        for layer in self.layers:
            stream = layer(stream)
        return self.readout(self.norm_out(stream[:, 0])), {}
