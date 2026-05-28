"""
Preprocess QALD-10 dataset into UnifiedSample JSONL files.

Reads raw QALD-10 JSON files (train.json, test.json) and converts
each sample into the unified format using Wikidata SPARQL queries.

Only English questions are extracted. SPARQL queries are normalized
from full URIs to the standard prefixed format (wd:, wdt:, p:, etc.)
for consistency with LC-QuAD 2.0 and SimpleQuestions.

Output layout:
    data/processed/qald10/
    └── qald10.jsonl        (all UnifiedSample records)
"""

from __future__ import annotations

import json
import re
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

RAW_DIR = Path("data/raw/qald10")
OUT_DIR = Path("data/processed/qald10")

# Official Wikidata SPARQL prefixes (longer URIs first to avoid partial matches)
# Please refer to https://www.wikidata.org/wiki/EntitySchema:E49
_URI_TO_PREFIX = [
    # Wikidata property namespaces (longest first within each group)
    ("http://www.wikidata.org/prop/statement/value-normalized/", "psn:"),
    ("http://www.wikidata.org/prop/statement/value/", "psv:"),
    ("http://www.wikidata.org/prop/statement/", "ps:"),
    ("http://www.wikidata.org/prop/qualifier/value-normalized/", "pqn:"),
    ("http://www.wikidata.org/prop/qualifier/value/", "pqv:"),
    ("http://www.wikidata.org/prop/qualifier/", "pq:"),
    ("http://www.wikidata.org/prop/reference/value-normalized/", "prn:"),
    ("http://www.wikidata.org/prop/reference/value/", "prv:"),
    ("http://www.wikidata.org/prop/reference/", "pr:"),
    ("http://www.wikidata.org/prop/direct-normalized/", "wdtn:"),
    ("http://www.wikidata.org/prop/direct/", "wdt:"),
    ("http://www.wikidata.org/prop/novalue/", "wdno:"),
    ("http://www.wikidata.org/prop/", "p:"),
    # Wikidata entity / value namespaces
    ("http://www.wikidata.org/wiki/Special:EntityData/", "wdata:"),
    ("http://www.wikidata.org/entity/statement/", "wds:"),
    ("http://www.wikidata.org/entity/", "wd:"),
    ("http://www.wikidata.org/reference/", "wdref:"),
    ("http://www.wikidata.org/value/", "wdv:"),
    # Wikidata service / ontology
    ("http://wikiba.se/ontology#", "wikibase:"),
    ("http://www.bigdata.com/rdf#", "bd:"),
    ("https://query.wikidata.org/subgraph/", "wdsubgraph:"),
    ("https://www.mediawiki.org/ontology#API/", "mwapi:"),
    # W3C standard namespaces
    ("http://www.w3.org/ns/lemon/ontolex#", "ontolex:"),
    ("http://www.w3.org/ns/prov#", "prov:"),
    ("http://www.w3.org/2004/02/skos/core#", "skos:"),
    ("http://www.w3.org/2002/07/owl#", "owl:"),
    ("http://www.w3.org/2001/XMLSchema#", "xsd:"),
    ("http://www.w3.org/2000/01/rdf-schema#", "rdfs:"),
    ("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf:"),
    # Other standard namespaces
    ("http://creativecommons.org/ns#", "cc:"),
    ("http://purl.org/dc/terms/", "dct:"),
    ("http://www.opengis.net/ont/geosparql#", "geo:"),
    ("http://schema.org/", "schema:"),
]

# Regex to match PREFIX declarations
_PREFIX_RE = re.compile(r"PREFIX\s+\w*:\s*<[^>]+>\s*", re.IGNORECASE)

# Regex to match full URIs in angle brackets: <http://...>
_URI_RE = re.compile(r"<(http[^>]+)>")

# Wikidata entity URI prefix for extracting entity IDs from answers
_WD_ENTITY_PREFIX = "http://www.wikidata.org/entity/"


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_sparql(sparql: str) -> str:
    """Normalize QALD-10 SPARQL from full URIs to standard prefixed format.

    1. Strip PREFIX declarations
    2. Replace <http://www.wikidata.org/...> URIs with wd:/wdt:/p:/etc. prefixes
    3. Collapse whitespace
    """
    # Strip PREFIX declarations
    result = _PREFIX_RE.sub("", sparql)

    # Replace full URIs with prefixed forms
    def replace_uri(match: re.Match) -> str:
        uri = match.group(1)
        for full, prefix in _URI_TO_PREFIX:
            if uri.startswith(full):
                return prefix + uri[len(full):]
        # Keep as full URI if no matching prefix
        return match.group(0)

    result = _URI_RE.sub(replace_uri, result)

    # Collapse whitespace
    return " ".join(result.split())


def extract_english_question(question_list: list[dict]) -> str:
    """Extract the first English question string."""
    for q in question_list:
        if q["language"] == "en":
            return q["string"]
    raise ValueError(f"No English question found: {question_list}")


def extract_gold_answer(answers: list[dict]) -> list[str]:
    """Extract gold answers as a flat list of strings.

    - URI answers: extract Wikidata entity ID (e.g. Q2212)
    - Literal answers: keep the literal value as-is
    - Boolean answers: store as string "true" or "false"
    """
    if not answers:
        return []

    answer_obj = answers[0]

    # Boolean answers (ASK queries)
    if "boolean" in answer_obj:
        return [str(answer_obj["boolean"]).lower()]

    # Binding-based answers (SELECT queries)
    bindings = answer_obj.get("results", {}).get("bindings", [])
    result: list[str] = []
    for binding in bindings:
        for val in binding.values():
            if val["type"] == "uri":
                uri = val["value"]
                if uri.startswith(_WD_ENTITY_PREFIX):
                    result.append(uri[len(_WD_ENTITY_PREFIX):])
            else:
                result.append(val["value"])
    return result


def convert(
    items: list[dict],
    split: str,
    labels: dict[str, str],
    properties: dict[str, list[str]],
) -> list[UnifiedSample]:
    """Convert raw QALD-10 records into UnifiedSample instances."""
    samples: list[UnifiedSample] = []
    for item in items:
        query = normalize_sparql(item["query"]["sparql"])
        samples.append(
            UnifiedSample(
                id=f"qald10-{split}-{item['id']}",
                source="qald10",
                split=split,
                question=extract_english_question(item["question"]),
                route=RouteType.SPARQL,
                kb_id="wikidata",
                target_query=query,
                gold_answer=extract_gold_answer(item["answers"]),
                metadata={
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
    train = load_json(RAW_DIR / "train.json")["questions"]
    test = load_json(RAW_DIR / "test.json")["questions"]

    print(f"Loaded: train={len(train)}, test={len(test)}")

    # Fetch labels and Wikidata-linked properties for all Q/P-IDs referenced in the queries
    labels, properties = fetch_query_metadata(
        [normalize_sparql(item["query"]["sparql"]) for item in train + test]
    )

    # Convert to UnifiedSample
    train_samples = convert(train, "train", labels, properties)
    test_samples = convert(test, "test", labels, properties)
    all_samples = train_samples + test_samples

    # Save samples
    save_jsonl(all_samples, str(OUT_DIR / "qald10.jsonl"))

    print(f"\nTotal: {len(all_samples)} samples written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
