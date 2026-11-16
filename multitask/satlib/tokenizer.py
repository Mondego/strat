"""Boolean tokens: the formula with every literal evaluated under the given assignment.

tokenize_onehot gives the type ids of the hand-written Type of the paper, tokenize_learned_type those of
the learned Type; tokenize picks one of them by Type mode. Each returns token_ids (the type ids),
raw_values (the Data value of every token) and the mask.
"""
import torch

TOK_PAD = 0
TOK_LIT_T = 1
TOK_LIT_F = 2
TOK_OR = 3
TOK_AND = 4
TOK_CLAUSE_START = 5
TOK_CLAUSE_END = 6
TOK_EOS = 7
N_TOKEN_TYPES = 8
N_LEARNED_TYPES = N_TOKEN_TYPES


def onehot_table():
    """Hand-written Type of every tokenize_onehot id."""
    return torch.eye(N_TOKEN_TYPES)


def tokenize_onehot(clauses, assignment):
    """Type ids of the hand-written Type and the Data value of every token."""
    token_ids, raw_values = [], []

    def emit(token, value=0.0):
        token_ids.append(token)
        raw_values.append(value)

    for clause_index, clause in enumerate(clauses):
        if clause_index > 0:
            emit(TOK_AND)
        emit(TOK_CLAUSE_START)
        for literal_index, literal in enumerate(clause):
            if literal_index > 0:
                emit(TOK_OR)
            variable = abs(literal)
            value = assignment[variable] if variable < len(assignment) else False
            if value == (literal > 0):
                emit(TOK_LIT_T, +1.0)
            else:
                emit(TOK_LIT_F, -1.0)
        emit(TOK_CLAUSE_END)
    emit(TOK_EOS)
    return {
        "token_ids": torch.tensor(token_ids, dtype=torch.long),
        "raw_values": torch.tensor(raw_values, dtype=torch.float),
        "mask": torch.ones(len(token_ids), dtype=torch.float),
    }


def tokenize_learned_type(clauses, assignment):
    """Type ids of the learned Type and the Data value of every token."""
    token_ids, raw_values = [], []

    def emit(token, value=0.0):
        token_ids.append(token)
        raw_values.append(value)

    for clause_index, clause in enumerate(clauses):
        if clause_index > 0:
            emit(TOK_AND)
        emit(TOK_CLAUSE_START)
        for literal_index, literal in enumerate(clause):
            if literal_index > 0:
                emit(TOK_OR)
            variable = abs(literal)
            value = assignment[variable] if variable < len(assignment) else False
            if value == (literal > 0):
                emit(TOK_LIT_T, +1.0)
            else:
                emit(TOK_LIT_F, -1.0)
        emit(TOK_CLAUSE_END)
    emit(TOK_EOS)
    return {
        "token_ids": torch.tensor(token_ids, dtype=torch.long),
        "raw_values": torch.tensor(raw_values, dtype=torch.float),
        "mask": torch.ones(len(token_ids), dtype=torch.float),
    }


def tokenize(clauses, assignment, type_mode='onehot'):
    """Return token_ids, raw_values and mask of one formula."""
    return {'onehot': tokenize_onehot, 'learned': tokenize_learned_type}[type_mode](clauses, assignment)


def data_inputs(raw_values):
    """Data: the value of every token and zero scratch fields."""
    d0 = raw_values.unsqueeze(-1)
    d1 = torch.zeros_like(d0)
    d2 = torch.zeros_like(d0)
    return torch.cat([d0, d1, d2], dim=-1)
