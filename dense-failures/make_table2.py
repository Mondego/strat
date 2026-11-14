"""
Builds the LaTeX for the paper's Table 2 (signed-sum ablation) from the ablation results:
error percentiles per configuration plus three measured mechanism columns.

Usage: python make_table2.py [--results results/ablation] [--out results/ablation/table2.tex]
Reads <results>/results.csv (run_ablation.py) and <results>/attention_test.csv
(attention_test.py). Writes the table to --out and prints it.
Requires booktabs and graphicx in the paper.

See make_table2.py.md.
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# (config key in the CSVs, label in the paper, strategy), in paper order.
ROWS = [
    ('A',      r'A',                 'Simplified Xformer'),
    ('As',     r'A$_s$',             'A Scaled'),
    ('Aln',    r'A$_{ln}$',          'LayerNorm'),
    ('Acurr',  r'A$_{curr}^\dagger$', 'Curriculum'),
    ('Bfl',    r'B$_{fl}$',          'Frozen Logic'),
    ('Bfa',    r'B$_{fa}$',          'Frozen Att'),
    ('Bortho', r'B$_{ortho}$',       'Regularized'),
    ('Cg',     r'C$_g$',             'Learned Gain'),
    ('Csb',    r'C$_{sb}$',          'Signal Boost'),
    ('Cc',     r'C$_c$',             'Reg + Boost'),
    ('Cb',     r'C$_b$',             'Balanced'),
    ('Cbi',    r'C$_{bi}$',          'Bias Init'),
]


def fmt_id(v):
    return f'{v:.1f}'


def fmt_ood(v):
    return f'{v:,.0f}'


def row_cells(res, att, key, label, strategy):
    g, a = res[res.Config == key], att[att.Config == key]
    idq = np.percentile(g.ID_MAE, [25, 50, 75])
    oodq = np.percentile(g.OOD_MAE, [50, 75])
    return [r'\textbf{' + label + '}', strategy, *map(fmt_id, idq), *map(fmt_ood, oodq),
            f'{a.id_flip_slope.median():.2f}', f'{a.ood_flip_none.median():.2f}',
            fmt_ood(a.ood_mae_gate.median())]


def build(res, att):
    seeds = sorted(res.Seed.unique())
    n = res.groupby('Config').size()
    assert n.nunique() == 1, f'unequal runs per config: {n.to_dict()}'
    lines = [
        r'\begin{table*}[!t]',
        r'\centering',
        r'\caption{Signed-sum ablation ($N{=}%d$ runs per configuration, seeds %d--%d). '
        r'MAE percentiles on fixed ID and OOD sets. Measured columns (medians; App.~\ref{app:measurements}): '
        r'\emph{ID slope}, the in-distribution effect of flipping one operator, regressed on the '
        r'flipped operand (1: proportional, as in the correct rule; 0: a fixed offset); '
        r'\emph{OOD flip}, the observed change in output when one operator is flipped, divided '
        r'by the correct change (1: correct; 0: operator ignored); \emph{Gate abl.}, OOD MAE when '
        r'the gate receives the input it would get if every operand were 50.5. '
        r'$^\dagger$A$_{curr}$ trains on operands up to 5,000, so 50.5 is not a typical operand '
        r'for it and its ID slope and gate ablation are not comparable with the other rows.}'
        % (n.iloc[0], seeds[0], seeds[-1]),
        r'\label{tab:failures}',
        r'\vspace{-0.1in}',
        r'\resizebox{\textwidth}{!}{%',
        r'{\scriptsize',
        r'\begin{tabular}{l l ccc cc ccc}',
        r'\toprule',
        r'& & \multicolumn{3}{c}{\textbf{ID MAE}} & \multicolumn{2}{c}{\textbf{OOD MAE}} '
        r'& \multicolumn{3}{c}{\textbf{Measured mechanism}} \\',
        r'\cmidrule(lr){3-5} \cmidrule(lr){6-7} \cmidrule(lr){8-10}',
        r'\textbf{Cfg} & \textbf{Strategy} & \textbf{25\%} & \textbf{Med} & \textbf{75\%} '
        r'& \textbf{Med} & \textbf{75\%} & \textbf{ID slope} & \textbf{OOD flip} & \textbf{Gate abl.} \\',
        r'\midrule',
        r'\textbf{Ref} & Hand-Coded & 0.0 & 0.0 & 0.0 & 0 & 0 & 1.00 & 1.00 & 0 \\',
        r'\midrule',
    ]
    for i, (key, label, strategy) in enumerate(ROWS):
        if i in (4, 7):
            lines.append(r'\midrule')
        lines.append(' & '.join(row_cells(res, att, key, label, strategy)) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}%', r'}}', r'\vspace{-0.1in}', r'\end{table*}']
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(HERE, 'results', 'ablation'))
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'ablation', 'table2.tex'))
    args = ap.parse_args()
    res = pd.read_csv(os.path.join(args.results, 'results.csv'))
    att = pd.read_csv(os.path.join(args.results, 'attention_test.csv'))
    tex = build(res, att)
    with open(args.out, 'w') as f:
        f.write(tex)
    print(tex)


if __name__ == '__main__':
    main()
