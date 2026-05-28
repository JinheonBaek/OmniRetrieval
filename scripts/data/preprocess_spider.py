"""
Preprocess Spider dataset into UnifiedSample JSONL files.

Reads raw Spider JSON files, merges them, shuffles with a fixed seed, 
and splits 70/30 into train and test sets.

The test_database directory (a superset of database/) is copied to the output
directory so that all referenced databases are available.

Output layout:
    data/processed/spider/
    ├── databases/          (copy of test_database/)
    ├── spider.jsonl        (all UnifiedSample records)
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

RAW_DIR = Path("data/raw/spider/spider_data")
OUT_DIR = Path("data/processed/spider")

SEED = 42
TRAIN_RATIO = 0.7


def load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def convert(items: list[dict], split: str) -> list[UnifiedSample]:
    """Convert raw Spider records into UnifiedSample instances."""
    samples: list[UnifiedSample] = []
    for idx, item in enumerate(items):
        samples.append(
            UnifiedSample(
                id=f"spider-{split}-{idx}",
                source="spider",
                split=split,
                question=item["question"],
                route=RouteType.SQL,
                kb_id=item["db_id"],
                target_query=item["query"],
                gold_answer=[],
                metadata={},
            )
        )
    return samples


def copy_databases(src: Path, dst: Path) -> None:
    """Copy the database directory to the output location."""
    if dst.exists():
        print(f"  Databases already exist: {dst}")
        return
    shutil.copytree(src, dst)
    print(f"  Copied databases -> {dst}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all splits
    train_spider = load_json(RAW_DIR / "train_spider.json")
    train_others = load_json(RAW_DIR / "train_others.json")
    dev = load_json(RAW_DIR / "dev.json")
    test = load_json(RAW_DIR / "test.json")

    print(f"Loaded: train_spider={len(train_spider)}, train_others={len(train_others)}, "
          f"dev={len(dev)}, test={len(test)}")

    # Merge, shuffle, and re-split
    merged = train_spider + train_others + dev + test
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
    save_jsonl(all_samples, str(OUT_DIR / "spider.jsonl"))

    # Save db_ids
    db_ids = sorted({s.kb_id for s in all_samples})
    db_ids_path = OUT_DIR / "db_ids.json"
    with open(db_ids_path, "w", encoding="utf-8") as f:
        json.dump(db_ids, f, indent=2)
    print(f"Saved {len(db_ids)} db_ids to {db_ids_path}")

    # Copy test_database (superset of database/)
    copy_databases(RAW_DIR / "test_database", OUT_DIR / "databases")

    print(f"\nTotal: {len(all_samples)} samples written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
