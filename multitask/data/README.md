# Datasets

This directory holds the data of the 11 dataset evaluations in the paper, one directory per dataset. Every dataset directory has two subdirectories: `id/` holds the in-distribution (ID) data, which are the training examples and the ID test set, and `ood/` holds the out-of-distribution (OOD) test set, which keeps the task rule. Each entry names the task and describes the files. Both models train on 10 base examples per dataset. The code of each dataset is in the task directory of the same name next to this one.

## SATNet Sudoku (`sudoku/`)

Constraint satisfaction (Sudoku solving): fill every empty cell of a 9×9 puzzle while preserving the given digits and the Sudoku constraints. `id/train_*.pt` and `id/test_*.pt` hold the 100 training and 1,000 test puzzles of SATNet Sudoku (Wang et al., 2019) with their solutions; the first 10 training puzzles are the training set, and the other 1,090 puzzles are the ID test set. `ood/sudoku.npz` holds 500 OOD puzzles with 35 clues, spread more unevenly over the row bands and column stacks than in any training puzzle.

## SATBench (`satbench/`)

Formula satisfaction classification (given-assignment satisfaction): given a Boolean formula in conjunctive normal form (CNF) and an assignment, predict whether the assignment satisfies every clause. `id/satbench_train.jsonl` holds the 10 satisfiable training formulas and `id/satbench_test.jsonl` the other 2,090 formulas of SATBench (Wei et al., 2025), 1,040 satisfiable and 1,050 unsatisfiable; each line stores the number of variables, the clauses, and whether the formula is satisfiable. `ood/satbench.json` holds 500 class-balanced test examples whose clause sequences are repeated cyclically to exactly 100 clauses.

## SATLIB (`satlib/`)

The same given-assignment satisfaction task as SATBench. `id/satlib_test.jsonl` holds 2,000 random 3-CNF formulas over 20 variables from SATLIB (Hoos and Stützle, 2000), 1,000 satisfiable with 91 clauses and 1,000 unsatisfiable with 110 clauses; the first 10 satisfiable formulas in file order are the training pool, and the other 1,990 formulas are the ID test set. `ood/satlib.json` holds 500 examples from 250 satisfiable test formulas extended to 100 clauses, each formula with one satisfying and one violating assignment.

## Tic-Tac-Toe Endgame (`tic_tac_toe/`)

Board classification (winner detection): predict whether X has completed a row, a column, or a diagonal. `id/tic-tac-toe.data` holds the 958 legal terminal boards of the UCI Tic-Tac-Toe Endgame dataset (Aha, 1991), one per line as nine cells followed by the label; the loader selects 10 base boards, expands them to 76 training boards with the eight symmetries of the square, and keeps 882 other boards as the ID test set. `ood/tic_tac_toe.json` holds 224 held-out boards, 112 per class, whose mark counts in the center, corners, and edges favor the opposite label under the correlations of the training set.

## Game of Life (`game_of_life/`)

State-transition prediction (next-state prediction): predict a 6×6 board with wrap-around boundaries after one update under the B3/S23 rule. `id/gol6x6.npz` holds 767 one-step transitions from the gameoflife-v1 collection (Strandgaard, 2023), boards in `X` and successors in `Y`; the first 10 pairs in file order are the training set, and the other 757 pairs are the ID test set. `ood/game_of_life.npz` holds the 500 generated OOD boards, each with exactly seven live cells and at most two adjacencies between live cells.

## PMLB Parity5 (`parity5/`)

Parity concept learning (five-bit parity): an input has ten bits, and the label is the parity of the first five; the other five are nuisance bits that share one value. `id/parity5.json` is built from the 32 assignments of PMLB Parity5 (Olson et al., 2017): its `train` split has 50 examples from 10 assignments and its `id` split has 160 examples over all 64 inputs, and in both the nuisance value agrees with the label in 80% of the examples. `ood/parity5.json` has the same 64 inputs with 20% agreement.

## PMLB M-of-N (`mofn/`)

Threshold concept learning (M-of-N rule): an input has 10 binary features, and the label is positive when at least three of seven designated features are active; the other three are nuisance features. `id/mofn.json` holds `train` (70 inputs: 10 base examples of the 1,324-row PMLB dataset (Olson et al., 2017) and the cyclic shifts of their relevant features) and `id` (the 1,231 held-out rows), and `id/pmlb/` holds the original PMLB file. `ood/mofn.json` holds `ood` (378 held-out inputs at the decision boundary: negatives with exactly two active relevant features and positives with exactly three) and three reference conditions.

## PMLB MONK-2 (`monk2/`)

Categorical rule induction (exact-two classification): an input has six categorical attributes, and the label is positive exactly when two of them take the distinguished value (raw value 1, stored as 0). `id/monk2.json` holds `train` (212 inputs: 10 base examples of the 601-row PMLB dataset (Olson et al., 2017) and their label-preserving augmentations) and `id` (303 held-out rows whose inputs do not occur in `train`). `ood/monk2.json` holds the 220 distinct `id` inputs with `weights` that raise the probability of category-count compositions absent from `train` from 18.48% to 32.70%, so OOD accuracy is weighted.

## PMLB LED24 (`led24/`)

Density classification (LED segment density): an input has 24 binary features, seven display segments followed by 17 nuisance features, and the label is positive when at least six segments are active (stored as label 0). `id/led24.json` holds `train` (140 inputs: 10 examples of the 3,200-row PMLB dataset (Olson et al., 2017) and their label-preserving augmentations) and `id` (the other 3,190 examples). `ood/led24.json` holds 500 inputs at the decision boundary with extreme nuisance counts: positives with six active segments and 15 active nuisance features, negatives with five active segments and at most one active nuisance feature.

## PMLB ThreeOf9 (`threeof9/`)

Majority classification (nine-bit majority): the label is positive when at least five of the nine input bits are active (stored as label 0). `id/threeof9.json` holds the 512 assignments of PMLB ThreeOf9 (Olson et al., 2017) labeled by this rule: `train` has 10 assignments and `id` has the other 502. `ood/threeof9.json` has the 144 assignments on which a shortcut that reads only bits 4, 6, and 9 predicts the wrong label.

## SPECT Heart (`spect/`)

Density classification (SPECT feature density): the input is the first nine binary features of the UCI SPECT Heart dataset (Cios et al., 2001), and the label is positive when at least four of them are active (stored as label 0). `id/spect.json` holds `train` (10 rows of the official training split) and `id` (163 rows of the official test split whose inputs do not occur in the training set). `ood/spect.json` holds 259 nine-bit inputs that never occur in the source dataset, obtained by cyclic rotations of the observed inputs.
