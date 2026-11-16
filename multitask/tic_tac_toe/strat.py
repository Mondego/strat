"""Type-only-attention STRAT classifier for Tic-Tac-Toe."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import N_CLASSES, N_LEARNED_TYPES, onehot_table, tokenize
from .type_embedding import TypeEmbedding


@dataclass(frozen=True)
class StratCfg:
    n_layers: int = 2
    n_heads: int = 4
    d_ctrl: int = 32
    d_data: int = 32
    hidden: int = 64
    gate_bias: float = 2.0
    qk_scale: float = 1.0
    type_mode: str = "onehot"
    type_width: int = 5

    @property
    def d_head(self):
        return self.d_ctrl // self.n_heads


class STRATGLU(nn.Module):
    def __init__(self, cfg: StratCfg):
        super().__init__()
        self.gate = nn.Linear(cfg.d_ctrl, cfg.d_data)
        self.value_1 = nn.Linear(2 * cfg.d_data, cfg.hidden)
        self.value_mid = nn.Linear(cfg.hidden, cfg.hidden)
        self.value_2 = nn.Linear(cfg.hidden, cfg.d_data, bias=False)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, cfg.gate_bias)
        nn.init.zeros_(self.value_2.weight)

    def forward(self, type_state, routed, local_data):
        gate = torch.sigmoid(self.gate(type_state))
        value = F.gelu(self.value_1(torch.cat([routed, local_data], dim=-1)))
        value = F.gelu(self.value_mid(value))
        return gate * self.value_2(value)


class StratLayer(nn.Module):
    def __init__(self, cfg: StratCfg):
        super().__init__()
        self.cfg = cfg
        self.norm_ctrl = nn.LayerNorm(cfg.d_ctrl)
        self.norm_data = nn.LayerNorm(cfg.d_data)
        self.W_q = nn.Linear(cfg.d_ctrl, cfg.d_ctrl, bias=False)
        self.W_k = nn.Linear(cfg.d_ctrl, cfg.d_ctrl, bias=False)
        self.W_v_type = nn.Linear(cfg.d_ctrl, cfg.d_ctrl, bias=False)
        self.W_o_type = nn.Linear(cfg.d_ctrl, cfg.d_ctrl, bias=False)
        self.W_v_data = nn.Linear(cfg.d_data, cfg.d_data, bias=False)
        self.alu = STRATGLU(cfg)

        nn.init.eye_(self.W_q.weight)
        nn.init.eye_(self.W_k.weight)
        with torch.no_grad():
            self.W_q.weight.mul_(cfg.qk_scale)
            self.W_k.weight.mul_(cfg.qk_scale)
        nn.init.eye_(self.W_v_type.weight)
        nn.init.zeros_(self.W_o_type.weight)
        nn.init.eye_(self.W_v_data.weight)

    def forward(self, type_state, data):
        cfg = self.cfg
        batch, n_tokens, _ = type_state.shape
        shape = (batch, n_tokens, cfg.n_heads, cfg.d_head)
        type_norm = self.norm_ctrl(type_state)
        query = self.W_q(type_norm).view(*shape).transpose(1, 2)
        key = self.W_k(type_norm).view(*shape).transpose(1, 2)
        scores = query @ key.transpose(-2, -1) / math.sqrt(cfg.d_head)
        attention = torch.softmax(scores, dim=-1)

        type_value = self.W_v_type(type_norm).view(*shape).transpose(1, 2)
        type_routed = (attention @ type_value).transpose(1, 2).reshape(
            batch, n_tokens, cfg.d_ctrl
        )
        type_new = type_state + self.W_o_type(type_routed)

        # Each head routes its own Data register.
        data_norm = self.norm_data(data)
        data_value = self.W_v_data(data_norm).view(*shape).transpose(1, 2)
        routed = (attention @ data_value).transpose(1, 2).reshape(
            batch, n_tokens, cfg.d_data
        )
        data_new = data + self.alu(type_new, routed, data_norm)
        return type_new, data_new


class STRAT(nn.Module):
    def __init__(self, cfg: StratCfg):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([StratLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm_out = nn.LayerNorm(cfg.d_data)
        self.readout = nn.Linear(cfg.d_data, N_CLASSES)
        self.type_embedding = TypeEmbedding(cfg.type_mode, onehot_table(cfg.d_ctrl), N_LEARNED_TYPES,
                                            cfg.type_width, cfg.d_ctrl)

    def forward(self, values):
        type_ids, data = tokenize(values, self.cfg.d_data, self.cfg.n_heads, self.cfg.type_mode)
        type_state = self.type_embedding(type_ids)
        for layer in self.layers:
            type_state, data = layer(type_state, data)
        return self.readout(self.norm_out(data[:, 0])), {}
