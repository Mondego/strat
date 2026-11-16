"""mofn: Type-only STRAT over observed features."""


from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import N_CLASSES, N_LEARNED_TYPES, onehot_table, tokenize
from .type_embedding import TypeEmbedding


@dataclass(frozen=True)
class StratCfg:
    n_layers: int = 1
    n_heads: int = 4
    d_type: int = 32
    d_data: int = 32
    hidden: int = 68
    gate_bias: float = 1
    qk_scale: float = 4
    type_mode: str = 'onehot'
    type_width: int = 5


def _initialize_qk(query, key, cfg):
    for projection in (query, key):
        nn.init.eye_(projection.weight)
        with torch.no_grad():
            projection.weight.mul_(cfg.qk_scale)


class STRATGLU(nn.Module):
    """Type chooses the write gate; routed and local Data supply the value."""
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.d_type, cfg.d_data)
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
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.norm_ctrl = nn.LayerNorm(cfg.d_type)
        self.norm_data = nn.LayerNorm(cfg.d_data)
        self.W_q = nn.Linear(cfg.d_type, cfg.d_type, bias=False)
        self.W_k = nn.Linear(cfg.d_type, cfg.d_type, bias=False)
        self.W_v_type = nn.Linear(cfg.d_type, cfg.d_type, bias=False)
        self.W_o_type = nn.Linear(cfg.d_type, cfg.d_type, bias=False)
        self.W_v_data = nn.Linear(cfg.d_data, cfg.d_data, bias=False)
        self.alu = STRATGLU(cfg)
        _initialize_qk(self.W_q, self.W_k, cfg)
        nn.init.eye_(self.W_v_type.weight)
        nn.init.zeros_(self.W_o_type.weight)
        nn.init.eye_(self.W_v_data.weight)

    def routing_weights(self, type_state):
        cfg = self.cfg
        shape = (*type_state.shape[:2], cfg.n_heads, cfg.d_type // cfg.n_heads)
        normed = self.norm_ctrl(type_state)
        query = self.W_q(normed).view(*shape).transpose(1, 2)
        key = self.W_k(normed).view(*shape).transpose(1, 2)
        return torch.softmax(query @ key.transpose(-2, -1) / math.sqrt(shape[-1]), dim=-1)

    def forward(self, type_state, data):
        cfg = self.cfg
        batch, n_tokens, _ = type_state.shape
        shape = (batch, n_tokens, cfg.n_heads, cfg.d_type // cfg.n_heads)
        attention = self.routing_weights(type_state)
        type_value = self.W_v_type(self.norm_ctrl(type_state)).view(*shape).transpose(1, 2)
        type_routed = (attention @ type_value).transpose(1, 2).reshape(batch, n_tokens, cfg.d_type)
        type_new = type_state + self.W_o_type(type_routed)
        data_norm = self.norm_data(data)
        data_value = self.W_v_data(data_norm).view(*shape).transpose(1, 2)
        routed = (attention @ data_value).transpose(1, 2).reshape(batch, n_tokens, cfg.d_data)
        return type_new, data + self.alu(type_new, routed, data_norm)


class STRAT(nn.Module):
    method = "strat"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([StratLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm_out = nn.LayerNorm(cfg.d_data)
        self.readout = nn.Linear(cfg.d_data, N_CLASSES)
        self.type_embedding = TypeEmbedding(cfg.type_mode, onehot_table(), N_LEARNED_TYPES,
                                            cfg.type_width, cfg.d_type)

    def forward_states(self, type_state, data):
        for layer in self.layers:
            type_state, data = layer(type_state, data)
        return self.readout(self.norm_out(data[:, 0])), {}

    def initial_states(self, values):
        type_ids, data = tokenize(values, self.cfg.type_mode)
        return self.type_embedding(type_ids), data

    def forward(self, values):
        return self.forward_states(*self.initial_states(values))
