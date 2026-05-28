"""Dataset loading and sampling utilities for the pipeline."""

import json
import random
from collections import defaultdict
from pathlib import Path

from src.data.schema import RouteType, UnifiedSample


# ------------------------------------------------------------------
# Dataset registry
# Maps dataset name → JSONL path relative to data/processed/
# ------------------------------------------------------------------
DATASET_PATHS = {
    # SEARCH (BEIR)
    "nfcorpus":        "beir/nfcorpus.jsonl",
    "scifact":         "beir/scifact.jsonl",
    "fiqa":            "beir/fiqa.jsonl",
    "msmarco":         "beir/msmarco.jsonl",
    "fever":           "beir/fever.jsonl",
    "nq":              "beir/nq.jsonl",
    "hotpotqa":        "beir/hotpotqa.jsonl",
    # SQL
    "spider":          "spider/spider.jsonl",
    "bird":            "bird/bird.jsonl",
    # SPARQL
    "lcquad2":         "lcquad2/lcquad2.jsonl",
    "qald10":          "qald10/qald10.jsonl",
    "simplequestions": "simplequestions/simplequestions.jsonl",
    # CYPHER
    "text2cypher":     "text2cypher/text2cypher.jsonl",
}

# ------------------------------------------------------------------
# Demo queries (wrapped as UnifiedSample)
# ------------------------------------------------------------------
DEMO_QUERIES = [
    # SEARCH
    UnifiedSample(
        id="demo-0", source="nfcorpus", split="test",
        question="Cancer Risk From French Fries",
        route=RouteType.SEARCH, kb_id="nfcorpus",
    ),
    UnifiedSample(
        id="demo-1", source="scifact", split="test",
        question="Myelin sheaths play a role in action potential propagation.",
        route=RouteType.SEARCH, kb_id="scifact",
    ),
    UnifiedSample(
        id="demo-2", source="fiqa", split="test",
        question="Credit card closed. Effect on credit score?",
        route=RouteType.SEARCH, kb_id="fiqa",
    ),
    UnifiedSample(
        id="demo-3", source="msmarco", split="test",
        question="when did the holocaust start and end",
        route=RouteType.SEARCH, kb_id="msmarco",
    ),
    UnifiedSample(
        id="demo-4", source="fever", split="test",
        question="Trevor Noah was born in the 20th century.",
        route=RouteType.SEARCH, kb_id="fever",
    ),
    UnifiedSample(
        id="demo-5", source="nq", split="test",
        question="who was originally offered the role of pretty woman",
        route=RouteType.SEARCH, kb_id="nq",
    ),
    UnifiedSample(
        id="demo-6", source="hotpotqa", split="test",
        question="Which software application was originally made for the company that acquired RadioShack in 1963?",
        route=RouteType.SEARCH, kb_id="hotpotqa",
    ),
    # SQL
    UnifiedSample(
        id="demo-7", source="spider", split="test",
        question="How many books do we have?",
        route=RouteType.SQL, kb_id="book_1",
    ),
    UnifiedSample(
        id="demo-8", source="spider", split="test",
        question="What are all distinct countries where singers above age 20 are from?",
        route=RouteType.SQL, kb_id="concert_singer",
    ),
    UnifiedSample(
        id="demo-9", source="bird", split="test",
        question="How many online purchases did Ole Group make in May 2019?",
        route=RouteType.SQL, kb_id="regional_sales",
    ),
    UnifiedSample(
        id="demo-10", source="bird", split="test",
        question="Which team was the home team in the match of the Bundesliga division on 2020/10/2?",
        route=RouteType.SQL, kb_id="european_football_1",
    ),
    # SPARQL
    UnifiedSample(
        id="demo-11", source="lcquad2", split="test",
        question="How many parent astronomical bodies are there by Jupiter?",
        route=RouteType.SPARQL, kb_id="wikidata",
    ),
    UnifiedSample(
        id="demo-12", source="qald10", split="test",
        question="When was Athens founded?",
        route=RouteType.SPARQL, kb_id="wikidata",
    ),
    UnifiedSample(
        id="demo-13", source="simplequestions", split="test",
        question="Who discovered 20468 Petercook?",
        route=RouteType.SPARQL, kb_id="wikidata",
    ),
    # CYPHER
    UnifiedSample(
        id="demo-14", source="text2cypher", split="test",
        question="Find the top 5 movies with the most votes.",
        route=RouteType.CYPHER, kb_id="neo4jlabs_demo_db_movies",
    ),
    UnifiedSample(
        id="demo-15", source="text2cypher", split="test",
        question="Identify the 5 suppliers with the highest average unit price of products supplied.",
        route=RouteType.CYPHER, kb_id="neo4jlabs_demo_db_northwind",
    ),
]


def load_samples(
    data_dir: str,
    datasets: list[str],
    split: str | None = None,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[UnifiedSample]:
    """Load samples from the specified datasets.

    Args:
        data_dir: Root directory of processed data.
        datasets: Dataset names to load (keys of DATASET_PATHS).
        split: If given, keep only samples with this split (e.g. "test").
        max_samples: If given, randomly sample up to this many per dataset.
        seed: Random seed for subsampling.
    """
    rng = random.Random(seed)
    samples = []
    data_root = Path(data_dir)

    for name in datasets:
        path = data_root / DATASET_PATHS[name]
        pool = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                sample = UnifiedSample.from_dict(json.loads(line))
                if split is None or sample.split == split:
                    pool.append(sample)

        if max_samples is not None and len(pool) > max_samples:
            pool = rng.sample(pool, max_samples)

        samples.extend(pool)
        print(f"Loaded {name}: {len(pool)} samples" + (f" (split={split})" if split else ""))

    return samples


def balance_by_route(
    samples: list[UnifiedSample],
    max_per_route: int | None = None,
    seed: int = 42,
) -> list[UnifiedSample]:
    """Subsample so each retrieval paradigm has at most max_per_route samples.

    If max_per_route is None, cap each paradigm to the size of the smallest.
    """
    by_route: dict[RouteType, list[UnifiedSample]] = defaultdict(list)
    for s in samples:
        by_route[s.route].append(s)

    if not by_route:
        return []

    cap = max_per_route if max_per_route is not None else min(len(v) for v in by_route.values())

    rng = random.Random(seed)
    out: list[UnifiedSample] = []
    for route in sorted(by_route, key=lambda r: r.value):
        group = by_route[route]
        if len(group) > cap:
            group = rng.sample(group, cap)
        out.extend(group)
    return out
