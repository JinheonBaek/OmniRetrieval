#!/usr/bin/env bash
# Download all datasets into data/raw/.
# Usage:
#   bash scripts/data/download_all.sh               # download everything
#   bash scripts/data/download_all.sh beir bird     # download specific datasets

set -e

DATASETS=(beir bird spider lcquad2 qald10 simplequestions text2cypher)

if [ $# -gt 0 ]; then
    DATASETS=("$@")
fi

for dataset in "${DATASETS[@]}"; do
    echo "=== Downloading ${dataset} ==="
    python scripts/data/download_"${dataset}".py
    echo
done

echo "All done."
