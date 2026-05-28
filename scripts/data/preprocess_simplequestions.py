"""
Preprocess SimpleQuestions (Wikidata) dataset into UnifiedSample JSONL files.

Reads raw SimpleQuestions TSV files and converts each sample into the
unified format. SPARQL queries are constructed from the (subject, property,
object) triples using the standard Wikidata prefixed format (wd:, wdt:).

Each line in the raw TSV has 4 tab-separated columns:
    subject_entity  property  object_entity  question

Property prefixes:
    P  → forward relation:  SELECT ?obj WHERE { wd:SUBJ wdt:PXXX ?obj }
    R  → reverse relation:  SELECT ?subj WHERE { ?subj wdt:PXXX wd:SUBJ }

Output layout:
    data/processed/simplequestions/
    └── simplequestions.jsonl    (all UnifiedSample records)
"""

from __future__ import annotations

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

RAW_DIR = Path("data/raw/simplequestions")
OUT_DIR = Path("data/processed/simplequestions")


def load_tsv(path: Path) -> list[tuple[str, str, str, str]]:
    """Load a SimpleQuestions TSV file.

    Returns list of (subject, property, object, question) tuples.
    """
    rows: list[tuple[str, str, str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", maxsplit=3)
            rows.append((parts[0], parts[1], parts[2], parts[3]))
    return rows


def build_sparql(subject: str, prop: str) -> str:
    """Build a Wikidata SPARQL query from a (subject, property) pair.

    P-prefixed properties denote forward relations:
        SELECT ?obj WHERE { wd:SUBJ wdt:PXXX ?obj }
    R-prefixed properties denote reverse relations
    (subject R_xx object means the KG triple is object P_xx subject):
        SELECT ?subj WHERE { ?subj wdt:PXXX wd:SUBJ }
    """
    prop_id = prop[1:]  # strip P/R prefix to get the numeric part
    if prop.startswith("P"):
        return f"SELECT ?obj WHERE {{ wd:{subject} wdt:P{prop_id} ?obj }}"
    if prop.startswith("R"):
        return f"SELECT ?subj WHERE {{ ?subj wdt:P{prop_id} wd:{subject} }}"
    raise ValueError(f"Invalid property prefix in {prop}: expected 'P' or 'R'")


def extract_gold_answer(prop: str, obj: str) -> list[str]:
    """Extract the gold answer entity ID from the triple.

    The answer is always the object column:
    - For forward relations (P): the object is the direct answer.
    - For reverse relations (R): the object is also the answer, since
      the KG triple is reversed (subject R_xx object means object P_xx subject).
    """
    if prop.startswith(("P", "R")):
        return [obj]
    raise ValueError(f"Invalid property prefix in {prop}: expected 'P' or 'R'")


def convert(
    items: list[tuple[str, str, str, str]],
    split: str,
    labels: dict[str, str],
    properties: dict[str, list[str]],
) -> list[UnifiedSample]:
    """Convert raw SimpleQuestions records into UnifiedSample instances."""
    samples: list[UnifiedSample] = []
    for idx, (subject, prop, obj, question) in enumerate(items):
        query = build_sparql(subject, prop)
        samples.append(
            UnifiedSample(
                id=f"simplequestions-{split}-{idx}",
                source="simplequestions",
                split=split,
                question=question,
                route=RouteType.SPARQL,
                kb_id="wikidata",
                target_query=query,
                gold_answer=extract_gold_answer(prop, obj),
                metadata={
                    "subject": subject,
                    "property": prop,
                    "object": obj,
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
    train = load_tsv(RAW_DIR / "annotated_wd_data_train_answerable.txt")
    test = load_tsv(RAW_DIR / "annotated_wd_data_test_answerable.txt")

    print(f"Loaded: train={len(train)}, test={len(test)}")

    # Fetch labels and Wikidata-linked properties for all Q/P-IDs referenced in the queries
    labels, properties = fetch_query_metadata([build_sparql(s, p) for s, p, _, _ in train + test])

    # Convert to UnifiedSample
    train_samples = convert(train, "train", labels, properties)
    test_samples = convert(test, "test", labels, properties)
    all_samples = train_samples + test_samples

    # Save samples
    save_jsonl(all_samples, str(OUT_DIR / "simplequestions.jsonl"))

    print(f"\nTotal: {len(all_samples)} samples written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
