#!/usr/bin/env bash
# Preprocess all datasets from data/raw/ into data/processed/.
# Usage:
#   bash scripts/data/preprocess_all.sh               # preprocess everything
#   bash scripts/data/preprocess_all.sh beir bird     # preprocess specific datasets

set -e

DATASETS=(beir bird spider lcquad2 qald10 simplequestions text2cypher)

if [ $# -gt 0 ]; then
    DATASETS=("$@")
fi

for dataset in "${DATASETS[@]}"; do
    echo "=== Preprocessing ${dataset} ==="
    python scripts/data/preprocess_"${dataset}".py
    echo
done

echo "All done."
