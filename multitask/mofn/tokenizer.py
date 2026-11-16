"""mofn tokens: one classification token, then one token per observed bit.

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
CLS_ROLE = 0
GROUP_RELEVANT = 1
GROUP_NUISANCE = 2
IDENTITY_START = 3
VALUE_START = 13
N_LEARNED_TYPES = 3


def feature_structure():
    """Type of every feature without its observed value."""
    types = torch.zeros(N_FEATURES, N_TYPE)
    types[2:9, GROUP_RELEVANT] = 1.0
    types[[0, 1, 9], GROUP_NUISANCE] = 1.0
    types[:, IDENTITY_START:IDENTITY_START + N_FEATURES] = torch.eye(N_FEATURES)
    return types


def onehot_table():
    """Hand-written Type of every tokenize_onehot id."""
    rows = feature_structure().repeat_interleave(TYPE_VALUES, dim=0)
    rows[:, VALUE_START:VALUE_START + TYPE_VALUES] = torch.eye(TYPE_VALUES).repeat(N_FEATURES, 1)
    cls = torch.zeros(1, N_TYPE)
    cls[0, CLS_ROLE] = 1.0
    return torch.cat([cls, rows])


def tokenize_onehot(values):
    """Type ids of the hand-written Type, and Data."""
    feature = torch.arange(values.shape[1], device=values.device)
    feature_ids = 1 + TYPE_VALUES * feature + values
    type_ids = torch.cat([values.new_zeros(values.shape[0], 1), feature_ids], dim=1)
    categories = F.one_hot(values, N_VALUES).float()
    data = categories.new_zeros(values.shape[0], 1 + values.shape[1], N_DATA)
    data[:, 1:, :N_VALUES] = categories
    return type_ids, data


def tokenize_learned_type(values):
    """Type ids of the learned Type, and Data."""
    feature_ids = torch.full_like(values, 2)
    feature_ids[:, 2:9] = 1
    type_ids = torch.cat([values.new_zeros(values.shape[0], 1), feature_ids], dim=1)
    categories = F.one_hot(values, N_VALUES).float()
    data = categories.new_zeros(values.shape[0], 1 + values.shape[1], N_DATA)
    data[:, 1:, :N_VALUES] = categories
    return type_ids, data


def tokenize(values, type_mode='onehot'):
    """Return type ids [batch, tokens] and Data [batch, tokens, d_data]."""
    return {'onehot': tokenize_onehot, 'learned': tokenize_learned_type}[type_mode](values)
