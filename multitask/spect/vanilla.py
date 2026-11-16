"""spect: unityped Transformer over observed features."""


from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import N_CLASSES, N_LEARNED_TYPES, onehot_table, tokenize
from .type_embedding import TypeEmbedding


@dataclass(frozen=True)
class VanillaCfg:
    n_layers: int = 1
    n_heads: int = 4
    d_type: int = 32
    d_data: int = 32
    hidden: int = 68
    qk_scale: float = 4
    type_mode: str = 'onehot'
    type_width: int = 5


def _initialize_qk(query, key, cfg):
    for projection in (query, key):
        nn.init.xavier_uniform_(projection.weight)
        with torch.no_grad():
            projection.weight.mul_(cfg.qk_scale)


class TransformerLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        width = cfg.d_type + cfg.d_data
        self.norm_attention = nn.LayerNorm(width)
        self.W_q = nn.Linear(width, width, bias=False)
        self.W_k = nn.Linear(width, width, bias=False)
        self.W_v = nn.Linear(width, width, bias=False)
        self.W_o = nn.Linear(width, width, bias=False)
        self.norm_ffn = nn.LayerNorm(width)
        self.ffn_1 = nn.Linear(width, cfg.hidden)
        self.ffn_mid = nn.Linear(cfg.hidden, cfg.hidden)
        self.ffn_2 = nn.Linear(cfg.hidden, width)
        _initialize_qk(self.W_q, self.W_k, cfg)
        nn.init.eye_(self.W_v.weight)
        nn.init.zeros_(self.W_o.weight)
        nn.init.zeros_(self.ffn_2.weight)
        nn.init.zeros_(self.ffn_2.bias)

    def forward(self, stream):
        cfg = self.cfg
        batch, n_tokens, width = stream.shape
        shape = (batch, n_tokens, cfg.n_heads, width // cfg.n_heads)
        normed = self.norm_attention(stream)
        query = self.W_q(normed).view(*shape).transpose(1, 2)
        key = self.W_k(normed).view(*shape).transpose(1, 2)
        value = self.W_v(normed).view(*shape).transpose(1, 2)
        attention = torch.softmax(query @ key.transpose(-2, -1) / math.sqrt(shape[-1]), dim=-1)
        routed = (attention @ value).transpose(1, 2).reshape(batch, n_tokens, width)
        stream = stream + self.W_o(routed)
        hidden = F.gelu(self.ffn_1(self.norm_ffn(stream)))
        hidden = F.gelu(self.ffn_mid(hidden))
        return stream + self.ffn_2(hidden)


class Vanilla(nn.Module):
    method = "transformer"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([TransformerLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm_out = nn.LayerNorm(cfg.d_type + cfg.d_data)
        self.readout = nn.Linear(cfg.d_type + cfg.d_data, N_CLASSES)
        self.type_embedding = TypeEmbedding(cfg.type_mode, onehot_table(), N_LEARNED_TYPES,
                                            cfg.type_width, cfg.d_type)

    def forward_states(self, type_state, data):
        stream = torch.cat([data, type_state], dim=-1)
        for layer in self.layers:
            stream = layer(stream)
        return self.readout(self.norm_out(stream[:, 0])), {}

    def initial_states(self, values):
        type_ids, data = tokenize(values, self.cfg.type_mode)
        return self.type_embedding(type_ids), data

    def forward(self, values):
        return self.forward_states(*self.initial_states(values))
