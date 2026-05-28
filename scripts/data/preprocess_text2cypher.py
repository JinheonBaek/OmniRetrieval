"""
Preprocess Neo4j text2cypher dataset into UnifiedSample JSONL files.

Filters applied:
  - Drop rows with null `database_reference_alias` (not executable against a known DB).
  - Drop tail DBs with negligible test coverage:
      * neo4jlabs_demo_db_stackoverflow (2 train / 0 test)
      * neo4jlabs_demo_db_openstreetmap (10 train / 1 test)

Output layout:
    data/processed/text2cypher/
    ├── text2cypher.jsonl                       (UnifiedSample records, per-sample schema in metadata)
    ├── db_ids.json                             (sorted list of surviving DB IDs)
    └── databases/<db_id>/schema.txt            (majority-variant schema per DB, for source selection)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.schema import RouteType, UnifiedSample, save_jsonl

RAW_DIR = Path("data/raw/text2cypher/data")
OUT_DIR = Path("data/processed/text2cypher")

DROP_DBS = {
    "neo4jlabs_demo_db_stackoverflow",
    "neo4jlabs_demo_db_openstreetmap",
}


def load_split(split: str) -> pd.DataFrame:
    fname = f"{split}-00000-of-00001.parquet"
    df = pd.read_parquet(RAW_DIR / fname)
    df = df.dropna(subset=["database_reference_alias"])
    df = df[~df["database_reference_alias"].isin(DROP_DBS)]
    return df.reset_index(drop=True)


def convert(df: pd.DataFrame, split: str) -> list[UnifiedSample]:
    samples: list[UnifiedSample] = []
    for _, row in df.iterrows():
        samples.append(UnifiedSample(
            id=f"text2cypher-{split}-{row['instance_id']}",
            source="text2cypher",
            split=split,
            question=row["question"],
            route=RouteType.CYPHER,
            kb_id=row["database_reference_alias"],
            target_query=row["cypher"],
            gold_answer=[],
            metadata={
                "data_source": row["data_source"],
                "schema": row["schema"],
            },
        ))
    return samples


def write_db_schemas(df: pd.DataFrame, databases_dir: Path) -> None:
    """Write the majority-variant schema for each DB as databases/<db_id>/schema.txt."""
    for db_id, sub in df.groupby("database_reference_alias"):
        canonical = sub["schema"].mode().iloc[0]
        db_dir = databases_dir / str(db_id)
        db_dir.mkdir(parents=True, exist_ok=True)
        (db_dir / "schema.txt").write_text(canonical, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train = load_split("train")
    test = load_split("test")
    print(f"After filters: train={len(train)}, test={len(test)}")

    samples = convert(train, "train") + convert(test, "test")
    save_jsonl(samples, str(OUT_DIR / "text2cypher.jsonl"))

    db_ids = sorted({s.kb_id for s in samples})
    db_ids_path = OUT_DIR / "db_ids.json"
    with open(db_ids_path, "w", encoding="utf-8") as f:
        json.dump(db_ids, f, indent=2)
    print(f"Saved {len(db_ids)} db_ids to {db_ids_path}")

    write_db_schemas(pd.concat([train, test]), OUT_DIR / "databases")
    print(f"Saved per-DB schemas to {OUT_DIR / 'databases'}/")

    print(f"\nTotal: {len(samples)} samples written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
