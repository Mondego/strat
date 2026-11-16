from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from .tokenizer import N_DIGITS, N_LEARNED_TYPES, onehot_table, tokenize
from .type_embedding import TypeEmbedding


@dataclass
class StratCfg:
    n_layers: int = 16
    n_heads: int = 4
    d_ctrl: int = 64            # Type width
    d_data: int = 18            # Data width: digit channels and scratch
    val_hidden: int = 128       # ALU hidden width
    type_mode: str = 'onehot'
    type_width: int = 37        # width of the learned Type table

    @property
    def d_head(self):
        return self.d_ctrl // self.n_heads


class STRATGLU(nn.Module):
    """The gate reads Type; the value reads the routed and the local Data."""

    def __init__(self, d_ctrl, d_data, d_hidden):
        super().__init__()
        self.w_gate = nn.Linear(d_ctrl, d_data)
        self.w_val_1 = nn.Linear(2 * d_data, d_hidden)
        self.w_val_2 = nn.Linear(d_hidden, d_data, bias=False)
        self.gate_act = nn.Sigmoid()
        self.val_act = nn.GELU()

        nn.init.zeros_(self.w_gate.weight)
        nn.init.constant_(self.w_gate.bias, 2.0)     # the gate starts open
        nn.init.zeros_(self.w_val_2.weight)          # the Data update starts at zero

    def forward(self, x_ctrl, x_data):
        gate = self.gate_act(self.w_gate(x_ctrl))
        val = self.w_val_2(self.val_act(self.w_val_1(x_data)))
        return gate * val, gate


class SudokuStratLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_ctrl = cfg.d_ctrl
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head

        self.ln_ctrl = nn.LayerNorm(cfg.d_ctrl)
        self.ln_data = nn.LayerNorm(cfg.d_data)
        self.W_q = nn.Linear(cfg.d_ctrl, cfg.d_ctrl, bias=False)
        self.W_k = nn.Linear(cfg.d_ctrl, cfg.d_ctrl, bias=False)
        self.W_v = nn.Linear(cfg.d_ctrl, cfg.d_ctrl, bias=False)
        self.W_o = nn.Linear(cfg.d_ctrl, cfg.d_ctrl, bias=False)
        self.ffn = STRATGLU(cfg.d_ctrl, cfg.d_data, cfg.val_hidden)

        nn.init.eye_(self.W_q.weight)        # attention starts as Type self-similarity
        nn.init.eye_(self.W_k.weight)
        nn.init.eye_(self.W_v.weight)
        nn.init.zeros_(self.W_o.weight)      # the Type update starts at zero

    def qk(self, T_n, D_n):
        """Query and key read only Type."""
        return self.W_q(T_n), self.W_k(T_n)

    def forward(self, T, D):
        B, N, _ = T.shape
        T_n = self.ln_ctrl(T)
        D_n = self.ln_data(D)

        # Attention: Q, K and V are projected from Type only.
        Q, K = self.qk(T_n, D_n)
        shape = (B, N, self.n_heads, self.d_head)
        Q = Q.view(*shape).transpose(1, 2)
        K = K.view(*shape).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        attn_weights = torch.softmax(scores, dim=-1)          # [B, heads, N, N]
        self.last_attn = attn_weights.mean(dim=1).detach()    # kept for visualization

        V = self.W_v(T_n).view(*shape).transpose(1, 2)
        Z_ctrl = torch.matmul(attn_weights, V).transpose(1, 2).reshape(B, N, self.d_ctrl)
        T_new = T + self.W_o(Z_ctrl)

        # Data is routed with the same attention; the gate reads Type, the value reads Data.
        routed = torch.matmul(attn_weights.mean(dim=1), D_n)
        data_update, _ = self.ffn(T_new, torch.cat([routed, D_n], dim=-1))
        D_new = D + data_update

        return T_new, D_new


class SudokuStrat(nn.Module):
    LAYER = SudokuStratLayer

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([self.LAYER(cfg) for _ in range(cfg.n_layers)])
        self.ln_out = nn.LayerNorm(cfg.d_data)
        self.W_out = nn.Linear(cfg.d_data, N_DIGITS)      # the readout reads only Data
        self.type_embedding = TypeEmbedding(cfg.type_mode, onehot_table(), N_LEARNED_TYPES, cfg.type_width, cfg.d_ctrl)

    def forward(self, givens):
        """givens [B, cells, digits] -> (logits [B, cells, digits], unused zero)."""
        type_ids, D = tokenize(givens, self.cfg.d_data, self.cfg.type_mode)
        T = self.type_embedding(type_ids)

        for block in self.blocks:
            T, D = block(T, D)

        res = self.W_out(self.ln_out(D))
        return res, givens.new_zeros(())
