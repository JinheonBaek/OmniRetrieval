"""
Unified schema for the OmniRetrieval framework.

Pipeline:
    Source selection:   query → backend (SEARCH | SQL | SPARQL | CYPHER) + knowledge base
    Query formulation:  produce a native query grounded in the structural context
    Execution:          run the native query against the native execution engine
    Evidence selection: keep the executor outputs relevant to the question
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RouteType(str, Enum):
    SEARCH = "SEARCH"
    SQL    = "SQL"
    SPARQL = "SPARQL"
    CYPHER = "CYPHER"


@dataclass
class UnifiedSample:
    """
    A single query sample in the unified OmniRetrieval format.

    Covers all four backends:
        SEARCH  — document retrieval    (BEIR corpora)
        SQL     — text-to-SQL           (BIRD, Spider)
        SPARQL  — text-to-SPARQL        (LC-QuAD 2.0, QALD-10, SimpleQuestions)
        CYPHER  — text-to-Cypher        (Neo4j Labs demo databases)
    """

    # Identifiers
    id:         str         # unique sample ID
    source:     str         # source dataset, e.g. "nfcorpus", "bird", "lcquad2"
    split:      str         # train | dev | test

    # Source selection
    question:   str         # user question (natural language)
    route:      RouteType   # selected backend (paradigm)
    kb_id:      str         # target knowledge base
                            #   SEARCH → BEIR subset name (e.g. "nfcorpus")
                            #   SQL    → database name    (e.g. "concert_singer")
                            #   SPARQL → KG name          (e.g. "wikidata")
                            #   CYPHER → Neo4j DB alias   (e.g. "neo4jlabs_demo_db_movies")

    # Query formulation
    target_query: str | None = None                         # target query to formulate (SQL/SPARQL/Cypher), if available
    gold_answer: list[str] = field(default_factory=list)    # flexible gold answer container:
                                                            #   SEARCH  → list of relevant corpus IDs (from qrels)
                                                            #   SQL     → list of expected result values (if available)
                                                            #   SPARQL  → list of answer entity URIs / literals
                                                            #   CYPHER  → list of result rows (if available)

    # Catch-all for dataset-specific fields not covered by the above schema
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "split": self.split,
            "question": self.question,
            "route": self.route.value,  # store enum as its value (string)
            "kb_id": self.kb_id,
            "target_query": self.target_query,
            "gold_answer": self.gold_answer,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UnifiedSample:
        return cls(
            id=data["id"],
            source=data["source"],
            split=data["split"],
            question=data["question"],
            route=RouteType(data["route"]),  # convert string back to enum
            kb_id=data["kb_id"],
            target_query=data.get("target_query"),
            gold_answer=data.get("gold_answer", []),
            metadata=data.get("metadata", {}),
        )


# ----------------------------------------------------------------------
# Utility functions for saving/loading samples in JSONL format
# ----------------------------------------------------------------------

def save_jsonl(samples: list[UnifiedSample], file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict()) + "\n")
        print(f"Saved {len(samples)} samples to {file_path}")


def load_jsonl(file_path: str) -> list[UnifiedSample]:
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(UnifiedSample.from_dict(json.loads(line)))
    print(f"Loaded {len(samples)} samples from {file_path}")
    return samples
