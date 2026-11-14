"""
Builds the LaTeX for the paper's Table 2 (failure taxonomy) from the ablation results.

Usage: python make_table2.py [--results results/ablation/results.csv]
                             [--out results/ablation/table2.tex]
Reads the per-run CSV written by run_ablation.py. Writes the table to --out and prints it.
Requires \\usepackage{booktabs} in the paper.

See make_table2.py.md.
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# (config key in results.csv, label in paper, strategy, hypothesis, diagnosis), paper order.
# Strategy / hypothesis / diagnosis text is copied verbatim from the submitted Table 2.
ROWS = [
    ('A',      'A',                       'Simplified Xformer', 'Standard Training',  'Linear Trap'),
    ('As',     r'A\textsubscript{s}',     'A Scaled',           'Scale Invariance',   'Linear Trap'),
    ('Aln',    r'A\textsubscript{ln}',    'LayerNorm',          'Internal Stability', 'Feature Collapse'),
    ('Acurr',  r'A\textsubscript{curr}',  'Curriculum',         'Complexity Sched.',  'Linear Trap'),
    ('Bfl',    r'B\textsubscript{fl}',    'Frozen Logic',       'Sharp Gates',        'Gradient Wall'),
    ('Bfa',    r'B\textsubscript{fa}',    'Frozen Att',         'Perfect Routing',    'Magnitude Leak'),
    ('Bortho', r'B\textsubscript{ortho}', 'Regularized',        'Orthogonality',      'Linear Trap'),
    ('Cg',     r'C\textsubscript{g}',     'Learned Gain',       'Self-Amplification', 'Gradient Wall'),
    ('Csb',    r'C\textsubscript{sb}',    'Signal Boost',       r'Fixed Gain ($\gamma$)', 'Gradient Wall'),
    ('Cc',     r'B\textsubscript{c}',     'Reg + Boost',        'Synergy',            'Gradient Wall'),
    ('Cb',     r'C\textsubscript{b}',     'Balanced',           'Operating Point',    'Operating Point Overfit'),
    ('Cbi',    r'C\textsubscript{bi}',    'Bias Init',          'Gradient Flow',      'Open gate trap'),
]


def fmt_id(v):
    return f'{v:.1f}'


def fmt_ood(v):
    return f'{v:,.0f}'.replace(',', '{,}')


def build(df):
    seeds = sorted(df.Seed.unique())
    n = df.groupby('Config').size()
    assert n.nunique() == 1, f'unequal runs per config: {n.to_dict()}'
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{The Failure Taxonomy ($N{=}%d$ runs/config, seeds %d--%d). We report '
        r'25\%%-ile / Median / 75\%%-ile errors.}' % (n.iloc[0], seeds[0], seeds[-1]),
        r'\label{tab:taxonomy}',
        r'\resizebox{\linewidth}{!}{%',
        r'\begin{tabular}{lll rrr rr l}',
        r'\toprule',
        r' & & & \multicolumn{3}{c}{ID MAE (Train)} & \multicolumn{2}{c}{OOD MAE (Test)} & \\',
        r'\cmidrule(lr){4-6} \cmidrule(lr){7-8}',
        r'Cfg & Strategy & Hypothesis & 25\% & Med & 75\% & Med & 75\% & Diagnosis / Mechanism \\',
        r'\midrule',
        r'Ref & Hand-Coded & Existence Proof & 0.0 & 0.0 & 0.0 & 0 & 0 & Geometric Logic \\',
        r'\midrule',
    ]
    for key, label, strategy, hypothesis, diagnosis in ROWS:
        g = df[df.Config == key]
        idq = np.percentile(g.ID_MAE, [25, 50, 75])
        oodq = np.percentile(g.OOD_MAE, [50, 75])
        cells = [label, strategy, hypothesis, *map(fmt_id, idq), *map(fmt_ood, oodq), diagnosis]
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(HERE, 'results', 'ablation', 'results.csv'))
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'ablation', 'table2.tex'))
    args = ap.parse_args()
    tex = build(pd.read_csv(args.results))
    with open(args.out, 'w') as f:
        f.write(tex)
    print(tex)


if __name__ == '__main__':
    main()
