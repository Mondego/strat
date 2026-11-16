"""Boolean Transformer with the shared inputs and full-stream attention/FFN."""

import math
import torch
import torch.nn as nn

from .tokenizer import N_LEARNED_TYPES, data_inputs, onehot_table
from .type_embedding import TypeEmbedding


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, alibi_slope):
        super().__init__()
        self.d_model = d_model
        self.alibi_slope = alibi_slope
        self.ln1 = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardBlock(d_model, 32)

    def forward(self, x, attention_mask):
        seq_len = x.shape[1]

        # Attention and the FFN read and update the whole stream.
        h = self.ln1(x)

        Q = self.W_q(h)
        K = self.W_k(h)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)

        # ALiBi, with the same slopes as STRAT.
        indices = torch.arange(seq_len, device=x.device)
        dist = indices.view(-1, 1) - indices.view(1, -1)
        scores = scores - self.alibi_slope * dist.float().abs()

        # Bidirectional attention; PAD keys are masked.
        scores = scores.masked_fill(~attention_mask[:, None, :], float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        # An entirely masked sample has no readable keys; replace its NaNs with zero.
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        V = self.W_v(h)
        Z = torch.matmul(attn_weights, V)

        x = x + self.W_o(Z)

        x = x + self.ffn(self.ln2(x))
        return x


class Vanilla(nn.Module):
    def __init__(self, n_layers=4, alibi_slopes=(1.0, 0.5, 0.0, 0.0), d_model=11,
                 type_mode='onehot', type_width=5):
        super().__init__()
        self.d_model = d_model

        self.blocks = nn.ModuleList([
            TransformerLayer(d_model, alibi_slope=alibi_slopes[i])
            for i in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.W_out = nn.Linear(d_model, 1)
        nn.init.xavier_uniform_(self.W_out.weight)
        nn.init.zeros_(self.W_out.bias)
        self.type_embedding = TypeEmbedding(type_mode, onehot_table(), N_LEARNED_TYPES, type_width, d_model - 3)

    def forward(self, token_ids, raw_values, mask):
        # The same [Data | Type] stream as STRAT.
        x = torch.cat([data_inputs(raw_values), self.type_embedding(token_ids)], dim=-1)
        attention_mask = mask > 0.5          # [B, S], True for real tokens

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        # Readout: sum over the real tokens.
        x = self.ln_final(x)
        x = x * mask.unsqueeze(-1)
        pooled = x.sum(dim=1)

        res = self.W_out(pooled)
        return res.squeeze(-1)
