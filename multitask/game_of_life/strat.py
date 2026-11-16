"""Type-only-attention STRAT for Game of Life."""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import N_LEARNED_TYPES, onehot_table, tokenize
from .topology import D4RelativeBias
from .type_embedding import TypeEmbedding


@dataclass(frozen=True)
class StratCfg:
    height: int = 6
    width: int = 6
    n_heads: int = 4
    d_ctrl: int = 32
    d_data: int = 32
    hidden: int = 68
    type_mode: str = "onehot"
    type_width: int = 5

    @property
    def d_head(self):
        return self.d_ctrl // self.n_heads


class STRATGLU(nn.Module):
    """Type chooses the write gate; routed and local Data supply the value."""

    def __init__(self, cfg: StratCfg):
        super().__init__()
        self.gate = nn.Linear(cfg.d_ctrl, cfg.d_data)
        self.value_1 = nn.Linear(2 * cfg.d_data, cfg.hidden)
        self.value_mid = nn.Linear(cfg.hidden, cfg.hidden)
        self.value_2 = nn.Linear(cfg.hidden, cfg.d_data, bias=False)

        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 1.0)
        nn.init.zeros_(self.value_2.weight)

    def forward(self, type_gate, routed, data):
        gate = torch.sigmoid(self.gate(type_gate))
        value = F.gelu(self.value_1(torch.cat([routed, data], dim=-1)))
        value = F.gelu(self.value_mid(value))
        value = self.value_2(value)
        return gate * value, gate


class STRAT(nn.Module):
    """Type-based routing followed by a gated local Data update."""

    def __init__(self, cfg: StratCfg | None = None):
        super().__init__()
        self.cfg = cfg or StratCfg()
        cfg = self.cfg
        self.n_tokens = cfg.height * cfg.width

        # Type and Data are normalized separately.
        self.norm_ctrl = nn.LayerNorm(cfg.d_ctrl)
        self.norm_data = nn.LayerNorm(cfg.d_data)
        self.W_q = nn.Linear(cfg.d_ctrl, cfg.d_ctrl)
        self.W_k = nn.Linear(cfg.d_ctrl, cfg.d_ctrl)
        self.W_v_type = nn.Linear(cfg.d_ctrl, cfg.d_ctrl)
        self.W_o_type = nn.Linear(cfg.d_ctrl, cfg.d_ctrl)
        self.W_v_data = nn.Linear(cfg.d_data, cfg.d_data)
        self.relative_bias = D4RelativeBias(cfg.n_heads, cfg.height, cfg.width)
        self.alu = STRATGLU(cfg)
        self.norm_out = nn.LayerNorm(cfg.d_data)
        self.readout = nn.Linear(cfg.d_data, 1)

        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.zeros_(self.W_q.bias)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.zeros_(self.W_k.bias)
        nn.init.eye_(self.W_v_type.weight)
        nn.init.zeros_(self.W_v_type.bias)
        nn.init.zeros_(self.W_o_type.weight)
        nn.init.zeros_(self.W_o_type.bias)
        nn.init.eye_(self.W_v_data.weight)
        nn.init.zeros_(self.W_v_data.bias)

        self.type_embedding = TypeEmbedding(cfg.type_mode, onehot_table(), N_LEARNED_TYPES, cfg.type_width, cfg.d_ctrl)

    def forward(self, grid):
        cfg = self.cfg
        type_ids, data = tokenize(grid, cfg.d_data, cfg.type_mode)
        type_state = self.type_embedding(type_ids)
        batch = grid.shape[0]
        shape = (batch, self.n_tokens, cfg.n_heads, cfg.d_head)

        # Q, K and V read only Type, and attention updates only Type.
        type_norm = self.norm_ctrl(type_state)
        query = self.W_q(type_norm).view(*shape).transpose(1, 2)
        key = self.W_k(type_norm).view(*shape).transpose(1, 2)
        type_value = self.W_v_type(type_norm).view(*shape).transpose(1, 2)
        scores = query @ key.transpose(-2, -1) / math.sqrt(cfg.d_head)
        attention = torch.softmax(scores + self.relative_bias().unsqueeze(0), dim=-1)
        type_routed = (attention @ type_value).transpose(1, 2).reshape(
            batch, self.n_tokens, cfg.d_ctrl
        )
        type_updated = type_state + self.W_o_type(type_routed)

        # Data is routed with the Type attention and written only by the gated ALU.
        data_norm = self.norm_data(data)
        value = self.W_v_data(data_norm).view(*shape).transpose(1, 2)
        routed = (attention @ value).transpose(1, 2).reshape(batch, self.n_tokens, cfg.d_data)

        update, _ = self.alu(type_updated, routed, data_norm)
        logits = self.readout(self.norm_out(data + update)).squeeze(-1)
        return logits, {}
