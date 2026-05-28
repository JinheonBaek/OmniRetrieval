"""
Preprocess LC-QuAD 2.0 dataset into UnifiedSample JSONL files.

Reads raw LC-QuAD 2.0 JSON files (train.json, test.json) and converts
each sample into the unified format using Wikidata SPARQL queries.

Output layout:
    data/processed/lcquad2/
    └── lcquad2.jsonl       (all UnifiedSample records)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data.wikidata_labels import (
    extract_entity_relations,
    extract_topic_entities,
    extract_topic_relations,
    fetch_query_metadata,
)
from src.data.schema import RouteType, UnifiedSample, save_jsonl

RAW_DIR = Path("data/raw/lcquad2")
OUT_DIR = Path("data/processed/lcquad2")


def load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_sparql(sparql: str) -> str:
    """Normalize Wikidata SPARQL to a consistent prefixed format.

    LC-QuAD 2.0 already uses wd:/wdt: prefixes, so only light cleanup
    is needed: strip leading/trailing whitespace and collapse internal
    whitespace to single spaces.
    """
    return " ".join(sparql.split())


def convert(
    items: list[dict],
    split: str,
    labels: dict[str, str],
    properties: dict[str, list[str]],
) -> list[UnifiedSample]:
    """Convert raw LC-QuAD 2.0 records into UnifiedSample instances."""
    samples: list[UnifiedSample] = []
    for item in items:
        query = normalize_sparql(item["sparql_wikidata"])
        samples.append(
            UnifiedSample(
                id=f"lcquad2-{split}-{item['uid']}",
                source="lcquad2",
                split=split,
                question=item["question"],
                route=RouteType.SPARQL,
                kb_id="wikidata",
                target_query=query,
                gold_answer=item["answer"],
                metadata={
                    "template_id": item["template_id"],
                    "paraphrased_question": item["paraphrased_question"],
                    "topic_entities": extract_topic_entities(query, labels),
                    "topic_relations": extract_topic_relations(query, labels),
                    "entity_relations": extract_entity_relations(query, labels, properties),
                },
            )
        )
    return samples


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load original splits
    train = load_json(RAW_DIR / "train.json")
    test = load_json(RAW_DIR / "test.json")

    print(f"Loaded: train={len(train)}, test={len(test)}")

    # Fetch labels and Wikidata-linked properties for all Q/P-IDs referenced in the queries
    labels, properties = fetch_query_metadata(
        [normalize_sparql(item["sparql_wikidata"]) for item in train + test]
    )

    # Convert to UnifiedSample
    train_samples = convert(train, "train", labels, properties)
    test_samples = convert(test, "test", labels, properties)
    all_samples = train_samples + test_samples

    # Save samples
    save_jsonl(all_samples, str(OUT_DIR / "lcquad2.jsonl"))

    print(f"\nTotal: {len(all_samples)} samples written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
