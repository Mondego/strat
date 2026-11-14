"""
Assigns failure-mode labels to each ablation run from its measured probes, using
thresholds fixed in advance (the constants below), and summarises the share of seeds per
label for each configuration.

Usage: python label_runs.py [--results results/ablation]
Reads <results>/probes.csv (probes.py) and <results>/dynamics.csv (dynamics.py).
Writes <results>/labels.csv (one row per run, one boolean column per label) and
<results>/label_summary.csv (percentage of seeds per label, per configuration); prints the
summary.

The thresholds and their justification are in label_runs.py.md. Change them only with a
dated entry there.
"""
import argparse
import os

import pandas as pd

import ablation

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- thresholds (pre-registered; see label_runs.py.md) ----
SUCCESS_OOD_MAE = 100.0          # same "Logic Acc" threshold as Section 6
SIGN_IGNORED = 0.5               # |sign gap| below this: operator has little effect (ideal 2)
SIGN_USED = 1.0                  # sign gap at least this: operator clearly used
FLIP_IGNORED = 0.25              # |flip ratio| below this: flipping an operator barely moves y (ideal 1)
LEAK_FIX = 0.5                   # removing magnitude from the gate cuts OOD MAE to <= this fraction
GATE_OPEN = 0.5                  # effective gate above this after both operators
GATE_SAME = 0.1                  # |gate after + minus gate after -| below this
CLAMP_HARMLESS = 1.1             # clamping the gate raises ID MAE by at most 10%
ROUTE_FAILED = 0.5               # fewer than half the numbers attend most to their operator
SATURATED = 0.9                  # median share of saturated gate units over training
ATTN_GRAD_STARVED = 0.1          # median attention-gradient norm below this x Config A's median
COLLAPSE_SPREAD = 0.25           # OOD prediction spread below this fraction of the target spread
SPREAD_ID = 0.75                 # ... while ID spread is at least this fraction

LABELS = ['Success', 'Linear Trap', 'Magnitude Leak', 'Open Gate Trap', 'Gradient Wall',
          'Feature Collapse', 'Unclassified']


def dynamics_summary(dyn):
    """Per-run medians over training of attention-gradient norm and gate saturation."""
    s = dyn.groupby(['Config', 'Seed']).agg(attn_grad_median=('grad_attn', 'median'),
                                             saturated_median=('gate_saturated', 'median'))
    return s.reset_index()


def label(p, a_attn_grad):
    lab = {}
    lab['Success'] = p.ood_mae < SUCCESS_OOD_MAE
    lab['Linear Trap'] = (abs(p.id_sign_gap) < SIGN_IGNORED and abs(p.ood_sign_gap) < SIGN_IGNORED
                          and abs(p.id_flip_ratio) < FLIP_IGNORED)
    lab['Magnitude Leak'] = (p.id_sign_gap >= SIGN_USED and abs(p.ood_sign_gap) < SIGN_IGNORED
                             and p.ood_d0_total_mae <= LEAK_FIX * p.ood_mae)
    lab['Open Gate Trap'] = (p.id_gate_plus > GATE_OPEN and p.id_gate_minus > GATE_OPEN
                             and abs(p.id_gate_diff) < GATE_SAME
                             and p.id_clamp_mae <= CLAMP_HARMLESS * p.id_mae)
    lab['Gradient Wall'] = (pd.notna(p.attn_grad_median)
                            and p.id_route_argmax < ROUTE_FAILED
                            and p.saturated_median > SATURATED
                            and p.attn_grad_median < ATTN_GRAD_STARVED * a_attn_grad)
    lab['Feature Collapse'] = (p.ood_spread_ratio < COLLAPSE_SPREAD
                               and p.id_spread_ratio >= SPREAD_ID)
    lab['Unclassified'] = not any(lab.values())
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(HERE, 'results', 'ablation'))
    args = ap.parse_args()
    probes = pd.read_csv(os.path.join(args.results, 'probes.csv'))
    dyn = dynamics_summary(pd.read_csv(os.path.join(args.results, 'dynamics.csv')))
    df = probes.merge(dyn, on=['Config', 'Seed'], how='left')
    a_attn_grad = df.loc[df.Config == 'A', 'attn_grad_median'].median()

    labels = pd.DataFrame([label(p, a_attn_grad) for p in df.itertuples()])
    out = pd.concat([df[['Config', 'Seed']], labels], axis=1)
    out.to_csv(os.path.join(args.results, 'labels.csv'), index=False)

    order = [c for c in ablation.CONFIGS if c in set(probes.Config)]
    summary = (out.groupby('Config')[LABELS].mean() * 100).reindex(order).round(0)
    summary.to_csv(os.path.join(args.results, 'label_summary.csv'))
    pd.set_option('display.width', 160)
    print('% of seeds per label (a run can carry several labels):')
    print(summary.to_string())


if __name__ == '__main__':
    main()
