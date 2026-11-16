"""parity5 tokens: one classification token, then one token per observed bit.

tokenize_onehot gives the type ids of the hand-written Type of the paper, tokenize_learned_type those of
the learned Type; tokenize picks one of them by Type mode. Both give the same Data.
"""

import torch
import torch.nn.functional as F

N_FEATURES = 10
N_CLASSES = 2
N_TYPE = N_DATA = 32
N_VALUES = 7
TYPE_VALUES = 2
VALUE_START = 3
CLS_ROLE = 0
CLS_RELEVANT = 1
N_RELEVANT = 5
N_LEARNED_TYPES = 3


def onehot_table():
    """Hand-written Type of every tokenize_onehot id."""
    table = torch.zeros(1 + 2 * TYPE_VALUES, N_TYPE)
    table[0, [CLS_ROLE, CLS_RELEVANT]] = 1.0
    for group in range(2):
        for value in range(TYPE_VALUES):
            table[1 + TYPE_VALUES * group + value, [1 + group, VALUE_START + value]] = 1.0
    return table


def tokenize_onehot(values):
    """Type ids of the hand-written Type, and Data."""
    group = (torch.arange(values.shape[1], device=values.device) >= N_RELEVANT).long()
    feature_ids = 1 + TYPE_VALUES * group + values
    type_ids = torch.cat([values.new_zeros(values.shape[0], 1), feature_ids], dim=1)
    categories = F.one_hot(values, N_VALUES).float()
    data = categories.new_zeros(values.shape[0], 1 + values.shape[1], N_DATA)
    data[:, 1:, :N_VALUES] = categories
    return type_ids, data


def tokenize_learned_type(values):
    """Type ids of the learned Type, and Data."""
    feature_ids = torch.full_like(values, 2)
    feature_ids[:, :N_RELEVANT] = 1
    type_ids = torch.cat([values.new_zeros(values.shape[0], 1), feature_ids], dim=1)
    categories = F.one_hot(values, N_VALUES).float()
    data = categories.new_zeros(values.shape[0], 1 + values.shape[1], N_DATA)
    data[:, 1:, :N_VALUES] = categories
    return type_ids, data


def tokenize(values, type_mode='onehot'):
    """Return type ids [batch, tokens] and Data [batch, tokens, d_data]."""
    return {'onehot': tokenize_onehot, 'learned': tokenize_learned_type}[type_mode](values)
