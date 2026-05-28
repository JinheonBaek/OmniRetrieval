"""
Preprocess BIRD dataset into UnifiedSample JSONL files.

Reads raw BIRD JSON files (train.json, dev.json), merges them, shuffles
with a fixed seed, and splits 70/30 into train and test sets.

Databases from both train_databases/ and dev_databases/ (which are
non-overlapping) are merged into a single output directory.

Output layout:
    data/processed/bird/
    ├── databases/          (merged from train_databases + dev_databases)
    ├── bird.jsonl          (all UnifiedSample records)
    └── db_ids.json         (sorted list of all referenced db_ids)
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.schema import RouteType, UnifiedSample, save_jsonl

RAW_DIR = Path("data/raw/bird")
OUT_DIR = Path("data/processed/bird")

SEED = 42
TRAIN_RATIO = 0.7


def load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def convert(items: list[dict], split: str) -> list[UnifiedSample]:
    """Convert raw BIRD records into UnifiedSample instances."""
    samples: list[UnifiedSample] = []
    for idx, item in enumerate(items):
        samples.append(
            UnifiedSample(
                id=f"bird-{split}-{idx}",
                source="bird",
                split=split,
                question=item["question"],
                route=RouteType.SQL,
                kb_id=item["db_id"],
                target_query=item["SQL"],
                gold_answer=[],
                metadata={
                    k: item[k] for k in ("evidence", "difficulty") if k in item
                },
            )
        )
    return samples


def merge_databases(srcs: list[Path], dst: Path) -> None:
    """Merge train and dev databases into a single output directory."""
    if dst.exists():
        print(f"  Databases already exist: {dst}")
        return

    dst.mkdir(parents=True, exist_ok=True)

    for src_dir in srcs:
        for db_path in sorted(src_dir.iterdir()):
            if db_path.is_dir():
                shutil.copytree(db_path, dst / db_path.name)

    print(f"  Merged databases -> {dst}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all splits
    train = load_json(RAW_DIR / "train" / "train.json")
    dev = load_json(RAW_DIR / "dev_20240627" / "dev.json")

    print(f"Loaded: train={len(train)}, dev={len(dev)}")

    # Merge, shuffle, and re-split
    merged = train + dev
    random.seed(SEED)
    random.shuffle(merged)

    split_idx = int(len(merged) * TRAIN_RATIO)
    train_items = merged[:split_idx]
    test_items = merged[split_idx:]

    print(f"After shuffle: train={len(train_items)}, test={len(test_items)}")

    # Convert to UnifiedSample
    train_samples = convert(train_items, "train")
    test_samples = convert(test_items, "test")
    all_samples = train_samples + test_samples

    # Save samples
    save_jsonl(all_samples, str(OUT_DIR / "bird.jsonl"))

    # Save db_ids
    db_ids = sorted({s.kb_id for s in all_samples})
    db_ids_path = OUT_DIR / "db_ids.json"
    with open(db_ids_path, "w", encoding="utf-8") as f:
        json.dump(db_ids, f, indent=2)
    print(f"Saved {len(db_ids)} db_ids to {db_ids_path}")

    # Merge databases from both train and dev
    merge_databases([
        RAW_DIR / "train" / "train_databases",
        RAW_DIR / "dev_20240627" / "dev_databases"
    ], OUT_DIR / "databases")

    print(f"\nTotal: {len(all_samples)} samples written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
