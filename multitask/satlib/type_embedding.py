"""Type of a token: the row of its type id in a table, zero-padded to the Type width.

    onehot   the hand-written table of the tokenizer, fixed
    learned  one trainable row per type id
"""
import torch.nn as nn
import torch.nn.functional as F


class TypeEmbedding(nn.Module):
    def __init__(self, type_mode, onehot_table, n_learned, learned_width, d_type):
        super().__init__()
        self.d_type = d_type
        if type_mode == 'onehot':
            self.register_buffer('table', onehot_table.float(), persistent=False)
        else:
            self.table = nn.Embedding(n_learned, learned_width).weight
            nn.init.normal_(self.table, mean=0.0, std=0.02)

    def forward(self, type_ids):
        types = F.embedding(type_ids, self.table)
        return F.pad(types, (0, self.d_type - types.shape[-1]))
