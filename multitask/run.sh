#!/usr/bin/env bash
# Train STRAT and the Transformer on one task for every seed:
#   bash multitask/run.sh sudoku cuda:0
#   TYPE_MODE=learned bash multitask/run.sh parity5
set -euo pipefail
cd "$(dirname "$0")/.."

TASK=$1
DEVICE=${2:-cpu}
TYPE_MODE=${TYPE_MODE:-onehot}
PYTHON_BIN=${PYTHON:-python3}

for seed in {41..50}; do
  for method in strat transformer; do
    "$PYTHON_BIN" -m "multitask.$TASK.run" --method "$method" --type-mode "$TYPE_MODE" --seed "$seed" --device "$DEVICE"
  done
done
