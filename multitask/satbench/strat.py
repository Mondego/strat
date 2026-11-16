"""Boolean STRAT: Type attention, gated Data updates, and masked Data sum readout."""

import math
import torch
import torch.nn as nn

from .tokenizer import N_LEARNED_TYPES, data_inputs, onehot_table
from .type_embedding import TypeEmbedding


class STRATGLU(nn.Module):
    """Type-controlled gate with a Data update."""

    def __init__(self, n_ctrl, n_data):
        super().__init__()
        n_hidden = 32
        self.w_gate = nn.Linear(n_ctrl, n_data)
        self.w_val_1 = nn.Linear(n_data, n_hidden, bias=False)
        self.w_val_2 = nn.Linear(n_hidden, n_data, bias=False)
        self.gate_act = nn.Sigmoid()
        self.val_act = nn.ReLU()

        nn.init.constant_(self.w_gate.bias, 1.0)
        nn.init.xavier_uniform_(self.w_gate.weight, gain=0.5)
        nn.init.normal_(self.w_val_1.weight, std=0.01)
        nn.init.normal_(self.w_val_2.weight, std=0.01)

    def forward(self, x_ctrl, x_data):
        gate = self.gate_act(self.w_gate(x_ctrl))
        val = self.w_val_2(self.val_act(self.w_val_1(x_data)))

        return gate * val


class StratLayer(nn.Module):
    def __init__(self, idx_ctrl, idx_data, alibi_slope):
        super().__init__()
        self.idx_ctrl = idx_ctrl
        self.idx_data = idx_data
        self.alibi_slope = alibi_slope
        n_ctrl = len(idx_ctrl)
        n_data = len(idx_data)
        self.W_q = nn.Linear(n_ctrl, n_ctrl, bias=False)
        self.W_k = nn.Linear(n_ctrl, n_ctrl, bias=False)
        self.W_v = nn.Linear(n_ctrl, n_ctrl, bias=False)
        self.ffn = STRATGLU(n_ctrl, n_data)

    def forward(self, x, attention_mask):
        seq_len = x.shape[1]

        # Attention reads and updates only the Type channels.
        x_ctrl = x[:, :, self.idx_ctrl]

        Q = self.W_q(x_ctrl)
        K = self.W_k(x_ctrl)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(len(self.idx_ctrl))

        # ALiBi: the score drops with distance, with one slope per layer.
        indices = torch.arange(seq_len, device=x.device)
        dist = indices.view(-1, 1) - indices.view(1, -1)
        scores = scores - self.alibi_slope * dist.float().abs()

        # Bidirectional attention; PAD keys are masked.
        scores = scores.masked_fill(~attention_mask[:, None, :], float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        # An entirely masked sample has no readable keys; replace its NaNs with zero.
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        V = self.W_v(x_ctrl)
        Z_ctrl = torch.matmul(attn_weights, V)

        x_mixed = x.clone()
        x_mixed[:, :, self.idx_ctrl] = x[:, :, self.idx_ctrl] + Z_ctrl

        # The gate reads Type and the value reads Data.
        ctrl_after_attn = x_mixed[:, :, self.idx_ctrl]
        data_before = x_mixed[:, :, self.idx_data]
        data_update = self.ffn(ctrl_after_attn, data_before)

        x_final = x_mixed.clone()
        x_final[:, :, self.idx_data] = x_mixed[:, :, self.idx_data] + data_update

        return x_final


class STRAT(nn.Module):
    def __init__(self, n_layers=4, alibi_slopes=(1.0, 0.5, 0.0, 0.0), d_type=8, type_mode='onehot', type_width=5):
        super().__init__()
        self.idx_data = [0, 1, 2]
        self.idx_ctrl = list(range(3, 3 + d_type))
        self.d_model = 3 + d_type
        self.d_type = d_type

        self.blocks = nn.ModuleList([
            StratLayer(self.idx_ctrl, self.idx_data, alibi_slope=alibi_slopes[i])
            for i in range(n_layers)
        ])
        self.W_out = nn.Linear(len(self.idx_data), 1)
        nn.init.xavier_uniform_(self.W_out.weight)
        nn.init.zeros_(self.W_out.bias)
        self.type_embedding = TypeEmbedding(type_mode, onehot_table(), N_LEARNED_TYPES, type_width, d_type)

    def forward(self, token_ids, raw_values, mask):
        # The stream is [Data | Type].
        x = torch.cat([data_inputs(raw_values), self.type_embedding(token_ids)], dim=-1)
        attention_mask = mask > 0.5          # [B, S], True for real tokens

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        # Readout: sum the Data of the real tokens.
        x_data = x[:, :, self.idx_data]
        x_data = x_data * mask.unsqueeze(-1)
        pooled = x_data.sum(dim=1)

        res = self.W_out(pooled)
        return res.squeeze(-1)
