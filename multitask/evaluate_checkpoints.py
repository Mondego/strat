"""Evaluate the trained models on the ID and OOD data of their task.

Run from the repository root, after training with run.sh:

    python -m multitask.evaluate_checkpoints --type-mode onehot
    python -m multitask.evaluate_checkpoints --type-mode learned --tasks sudoku parity5

Each model is loaded with the code in its task directory. Prints one line per model, then the mean and
SEM over the seeds.
"""
import argparse
import importlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
METHODS = ('strat', 'transformer')
TASKS = ('sudoku', 'satbench', 'satlib', 'tic_tac_toe', 'game_of_life',
         'parity5', 'mofn', 'monk2', 'led24', 'threeof9', 'spect')
TABULAR = TASKS[5:]
SEEDS = range(41, 51)


def modules(task, *names):
    return [importlib.import_module(f'.{task}.{name}', __package__) for name in names]


def load(task, type_mode, method, seed):
    path = ROOT / task / 'results' / type_mode / f'{method}_seed{seed}.pt'
    return torch.load(path, map_location='cpu', weights_only=False)


def restore(model, state):
    model.load_state_dict(state)
    return model.eval()


def tabular(task, type_mode):
    run, strat, vanilla = modules(task, 'run', 'strat', 'vanilla')
    splits = run.load_splits()
    rows = []
    for method in METHODS:
        cls, cfg = (strat.STRAT, strat.StratCfg) if method == 'strat' else (vanilla.Vanilla, vanilla.VanillaCfg)
        for seed in SEEDS:
            saved = load(task, type_mode, method, seed)
            model = restore(cls(cfg(**saved['payload']['model_config'])), saved['model'])
            accuracy = {}
            for split in ('id', 'ood'):
                data = splits[split]
                values = torch.tensor(data['values'], dtype=torch.long)
                with torch.inference_mode():
                    logits = torch.cat([model(chunk)[0] for chunk in values.split(256)])
                weights = data.get('weights', [1.0] * len(data['labels']))
                correct = math.fsum(w * (p == y) for w, p, y in zip(weights, logits.argmax(-1).tolist(), data['labels']))
                accuracy[split] = correct / math.fsum(weights)
            rows.append({'method': method, 'seed': seed, **accuracy})
    return rows


def game_of_life(type_mode):
    run, strat, vanilla = modules('game_of_life', 'run', 'strat', 'vanilla')
    _, _, x_test, y_test = run.load_data()
    ood = np.load(DATA / 'game_of_life/ood/game_of_life.npz')
    x_ood, y_ood = torch.tensor(ood['X']).float(), torch.tensor(ood['Y']).float()
    rows = []
    for method in METHODS:
        cls, cfg = (strat.STRAT, strat.StratCfg) if method == 'strat' else (vanilla.Vanilla, vanilla.VanillaCfg)
        for seed in SEEDS:
            saved = load('game_of_life', type_mode, method, seed)
            model = restore(cls(cfg(**saved['architecture'])), saved['model'])
            rows.append({'method': method, 'seed': seed, 'id': run.evaluate(model, x_test, y_test),
                         'ood': run.evaluate(model, x_ood, y_ood)})
    return rows


def tic_tac_toe(type_mode):
    run, strat, vanilla = modules('tic_tac_toe', 'run', 'strat', 'vanilla')
    loaded = run.load_task()
    ood = json.load((DATA / 'tic_tac_toe/ood/tic_tac_toe.json').open())
    value_maps = [{value: index for index, value in enumerate(names)} for names in loaded.value_names]
    label_map = {label: index for index, label in enumerate(loaded.label_names)}
    ood_values = torch.tensor([[value_maps[c][str(v)] for c, v in enumerate(row)] for row in ood['raw_values']])
    ood_labels = torch.tensor([label_map[str(label)] for label in ood['raw_labels']])
    device = torch.device('cpu')
    rows = []
    for method in METHODS:
        cls, cfg = (strat.STRAT, strat.StratCfg) if method == 'strat' else (vanilla.Vanilla, vanilla.VanillaCfg)
        for seed in SEEDS:
            saved = load('tic_tac_toe', type_mode, method, seed)
            model = restore(cls(cfg(**saved['payload']['architecture'])), saved['model'])
            rows.append({'method': method, 'seed': seed,
                         'id': run.evaluate(model, loaded.test_values, loaded.test_labels, device),
                         'ood': run.evaluate(model, ood_values, ood_labels, device)})
    return rows


def boolean(task, type_mode):
    run, strat, vanilla = modules(task, 'run', 'strat', 'vanilla')
    _, test_instances = run.experiment_split()
    ood = json.load((DATA / task / 'ood' / f'{task}.json').open())
    raw_ood = [(s['clauses'], [False] + [bool(v) for v in s['assignment']], s['num_vars'], s['label']) for s in ood['samples']]
    batch_ood = run.collate(raw_ood, type_mode)
    device = torch.device('cpu')
    rows = []
    for method in METHODS:
        cls = strat.STRAT if method == 'strat' else vanilla.Vanilla
        for seed in SEEDS:
            saved = load(task, type_mode, method, seed)
            model = restore(cls(**saved['meta']), saved['model']).to(device)
            # The ID assignments of unsatisfiable formulas are drawn with the evaluation seed of the training run.
            batch_id = run.collate(run._full_test(test_instances, seed=seed + 1000), type_mode)
            rows.append({'method': method, 'seed': seed, 'id': run._accuracy(model, batch_id, device, chunk=64),
                         'ood': run._accuracy(model, batch_ood, device, chunk=32)})
    return rows


def sudoku(type_mode):
    run, strat, vanilla = modules('sudoku', 'run', 'strat', 'vanilla')
    _, _, givens_test, solutions_test = run.load_splits()
    ood = np.load(DATA / 'sudoku/ood/sudoku.npz')
    puzzles = torch.tensor(ood['puzzles'], dtype=torch.long).reshape(-1, 81)
    givens_ood = torch.zeros(len(puzzles), 81, 9)
    mask = puzzles > 0
    givens_ood[mask] = F.one_hot(puzzles[mask] - 1, 9).float()
    solutions_ood = torch.tensor(ood['solutions'], dtype=torch.long).reshape(-1, 81) - 1
    device = torch.device('cpu')
    rows = []
    for method in METHODS:
        cls, cfg = ((strat.SudokuStrat, strat.StratCfg) if method == 'strat'
                    else (vanilla.SudokuVanilla, vanilla.VanillaCfg))
        for seed in SEEDS:
            saved = load('sudoku', type_mode, method, seed)
            model = restore(cls(cfg(**saved['cfg'])), saved['model']).to(device)
            rows.append({'method': method, 'seed': seed,
                         'id': run.evaluate(model, givens_test, solutions_test, device, 128),
                         'ood': run.evaluate(model, givens_ood, solutions_ood, device, 128)})
    return rows


def evaluate(task, type_mode):
    """One row per model: its method, seed, ID accuracy and OOD accuracy."""
    if task in TABULAR:
        return tabular(task, type_mode)
    if task == 'game_of_life':
        return game_of_life(type_mode)
    if task == 'tic_tac_toe':
        return tic_tac_toe(type_mode)
    if task == 'sudoku':
        return sudoku(type_mode)
    return boolean(task, type_mode)


def mean_sem(values):
    return f'{100 * statistics.mean(values):6.2f} ± {100 * statistics.stdev(values) / len(values) ** .5:.2f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--type-mode', choices=('onehot', 'learned'), required=True)
    parser.add_argument('--tasks', choices=TASKS, nargs='+', default=list(TASKS))
    args = parser.parse_args()
    torch.set_num_threads(4)
    records = {}
    for task in args.tasks:
        records[task] = evaluate(task, args.type_mode)
        for r in records[task]:
            print(f"{task:14s} {r['method']:11s} seed={r['seed']} id={r['id']:.6f} ood={r['ood']:.6f}", flush=True)
    print(f'\nType mode {args.type_mode}: mean ± SEM over seeds (%)')
    for task, rows in records.items():
        for method in METHODS:
            selected = [r for r in rows if r['method'] == method]
            print(f"{task:14s} {method:11s} ID {mean_sem([r['id'] for r in selected])}   "
                  f"OOD {mean_sem([r['ood'] for r in selected])}   n={len(selected)}")


if __name__ == '__main__':
    main()
