# STRAT on diverse tasks

Code and data for the evaluation of STRAT across diverse tasks. STRAT is compared with a Transformer on 11 datasets that cover 10 tasks. Within each dataset the two models receive the same observable inputs and the same training data: 10 base examples and, where a task has them, the same augmentations. They also share depth, head count, hidden widths, optimizer, learning rate and training length. Results are the mean accuracy and its standard error over the paired seeds 41 to 50. SATNet Sudoku and Game of Life count a board as correct only when every cell is correct; the other tasks count single examples.

## Layout

```
multitask/
  run.sh                   train both models on one task for the seeds 41 to 50
  evaluate_checkpoints.py  ID and OOD accuracy of the trained models
  data/                    one directory per dataset, described in data/README.md
  <task>/                  one directory per dataset, named like its data directory
```

Every task directory has the same five files. `tokenizer.py` turns an input into type ids and Data, `type_embedding.py` turns a type id into the Type vector, `strat.py` and `vanilla.py` define STRAT and the Transformer, and `run.py` loads the data, trains and evaluates one model; it is the entry point. `game_of_life/topology.py` additionally holds the relative-position bias that both models use.

## Usage

Run from the repository root. The code needs Python 3.10 or later, PyTorch and NumPy; the SATBench and SATLIB runs also need `pycosat`.

```bash
# Train STRAT and the Transformer on one task for the seeds 41 to 50.
bash multitask/run.sh parity5
TYPE_MODE=learned bash multitask/run.sh sudoku cuda:0

# Train one model.
python -m multitask.parity5.run --method strat --type-mode onehot --seed 41

# ID and OOD accuracy of the trained models, mean and standard error over the seeds.
python -m multitask.evaluate_checkpoints --type-mode onehot
```

A run has five options: `--method` (`strat` or `transformer`), `--type-mode`, `--seed`, `--epochs` and `--device`. It saves the model of the last epoch to `multitask/<task>/results/<type_mode>/<method>_seed<seed>.pt`. `run.sh` calls `python3` unless the `PYTHON` variable names another interpreter. The evaluation needs both models of every seed for each task it covers, and `--tasks` limits it to the tasks that were trained.

## Type and Data

Every token has a Type vector and a Data vector. In STRAT, attention and the gates read only Type, and Data carries the values that are routed and updated. The Transformer receives the same Type and Data concatenated into one stream. The tokenizer of a task gives each token a type id and its Data, and the `type_embedding.py` of the task turns the type id into the Type vector, padded with zeros to the Type width. `--type-mode` selects how:

- `onehot` is the configuration of the paper. The type id selects a row of the hand-written Type table of the task, which the next section describes. The table is fixed and has no parameters.
- `learned` replaces the hand-written table with a trainable one: each type id has its own vector, initialized from N(0, 0.02). The type ids of this mode come from `tokenize_learned_type` in the tokenizer of each task.

## Tasks

All models are trained with gradient clipping at norm 1.0 and evaluated at the last epoch. `data/README.md` describes the training, ID and OOD data of each dataset.

### SATNet Sudoku (`sudoku/`)

Constraint satisfaction: fill every empty cell of a 9×9 puzzle. Each of the 81 cells is one token. Type has 64 channels, of which 37 encode the row, column and box of the cell, whether the cell is given, and the given digit; Data has 18 channels, a digit register and a candidate register. Both models have 16 four-head layers and train for 6,000 epochs.

### SATBench (`satbench/`)

Formula satisfaction classification: given a formula in conjunctive normal form and an assignment, predict whether the assignment satisfies every clause. The formula is serialized with eight token types: true literal, false literal, OR, AND, clause start, clause end, end of sequence and padding. Type is the one-hot of the token type, and Data has three channels, the first of which holds +1 for a true literal and −1 for a false literal. Both models have four single-head layers and train for 2,000 epochs.

### SATLIB (`satlib/`)

The same assignment-verification task on random 3-CNF formulas over 20 variables. The tokens, the model configuration and the training length are those of SATBench.

### Tic-Tac-Toe Endgame (`tic_tac_toe/`)

Board classification: predict whether X has completed a row, a column or a diagonal. A board is one classification token, nine cell tokens and eight line tokens for the three rows, three columns and two diagonals. The Type of a cell encodes its role, row, column, diagonal membership and mark, and the Type of a line identifies the line; the Data of a cell holds its mark. Both models have two four-head layers, read the prediction from the classification token and train for 2,000 epochs.

### Game of Life (`game_of_life/`)

State-transition prediction: predict a 6×6 board with wrap-around boundaries after one update under the B3/S23 rule. Each of the 36 cells is one token. Type stores the alive or dead input state as a two-channel one-hot, and Data holds the alive bit. Both models have one four-head layer and train for 6,000 epochs.

### Feature-classification tasks

The six tasks below share one layout. An input is one classification token followed by one token per feature, every token has 32 Type and 32 Data channels, Data holds the observed value of the feature and scratch channels, and the prediction is read from the classification token. Both models have four heads and train for 2,000 epochs.

**PMLB Parity5 (`parity5/`).** Parity concept learning: the label is the parity of five relevant bits, and five nuisance bits never affect it. Type marks whether a bit is relevant or nuisance and its observed value. Both models have two layers.

**PMLB M-of-N (`mofn/`).** Threshold concept learning: the label is positive when at least three of seven designated features are active. Type marks the relevant or nuisance group of a feature, its identity and its observed value. Both models have one layer.

**PMLB MONK-2 (`monk2/`).** Categorical rule induction: the label is positive exactly when two of six categorical attributes take their distinguished value. Type encodes the identity of an attribute and its observed category. Both models have one layer.

**PMLB LED24 (`led24/`).** Density classification: the label is positive when at least six of the seven display segments are active, and 17 nuisance features do not affect it. Type marks whether a feature is a segment or a nuisance feature, its identity and its observed value. Both models have one layer.

**PMLB ThreeOf9 (`threeof9/`).** Majority classification: the label is positive when at least five of nine bits are active. Type encodes the identity of a bit and its observed value. Both models have one layer.

**SPECT Heart (`spect/`).** Density classification: the label is positive when at least four of the first nine binary features are active. Type encodes the identity of a feature and its observed value. Both models have one layer.
