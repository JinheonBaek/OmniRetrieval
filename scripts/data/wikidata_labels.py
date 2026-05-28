"""Fetch English labels and entity-linked properties for Wikidata Q/P-IDs referenced in SPARQL queries."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import requests

API_URL = "https://www.wikidata.org/w/api.php"
BATCH = 50
SLEEP = 0.2
QID_RE = re.compile(r"\bwd:(Q\d+)\b")
PID_RE = re.compile(r"\b[a-z]+:(P\d+)\b")
LABELS_CACHE_PATH = Path("data/cache/wikidata/labels.json")
PROPERTIES_CACHE_PATH = Path("data/cache/wikidata/properties.json")


def _request(params: dict) -> dict:
    """GET the Wikidata API with retry-on-error and Retry-After honoring."""
    for attempt in range(10):
        try:
            r = requests.get(API_URL, params=params, timeout=30,
                             headers={"User-Agent": "OmniRetrieval/1.0 (https://github.com/JinheonBaek/OmniRetrieval)"})
            r.raise_for_status()
            return r.json().get("entities", {})
        except (requests.RequestException, ValueError) as e:
            response = getattr(e, "response", None)
            retry_after = response.headers.get("Retry-After", "") if response is not None else ""
            wait = int(retry_after) if retry_after.isdigit() else 2 ** attempt
            print(f"  retry {attempt + 1}/10 after {wait}s: {e}")
            time.sleep(wait)
    raise RuntimeError("Wikidata API failed after 10 retries")


def _fetch_labels(ids: list[str]) -> dict[str, str]:
    entities = _request({
        "action": "wbgetentities",
        "ids": "|".join(ids),
        "props": "labels",
        "languages": "en",
        "format": "json",
    })
    return {
        qid: entity.get("labels", {}).get("en", {}).get("value", "")
        for qid, entity in entities.items()
    }


def _fetch_properties(ids: list[str]) -> dict[str, list[str]]:
    entities = _request({
        "action": "wbgetentities",
        "ids": "|".join(ids),
        "props": "claims",
        "format": "json",
    })
    return {
        qid: list(entity.get("claims", {}).keys())
        for qid, entity in entities.items()
    }


def load_cache(path: Path) -> dict[str, Any]:
    """Load a JSON cache from disk, or return empty if the file is absent."""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    """Write a JSON cache to disk, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, sort_keys=True)


def get_labels(qids: set[str]) -> dict[str, str]:
    """Return {qid: english_label} for all requested Q/P-IDs, using an on-disk cache."""
    cache = load_cache(LABELS_CACHE_PATH)
    missing = sorted(q for q in qids if q not in cache)
    for i in range(0, len(missing), BATCH):
        chunk = missing[i : i + BATCH]
        cache.update(_fetch_labels(chunk))
        time.sleep(SLEEP)
        if i % (BATCH * 20) == 0:
            print(f"  fetched {i + len(chunk)}/{len(missing)} labels")
    save_cache(LABELS_CACHE_PATH, cache)
    return {q: cache.get(q, "") for q in qids}


def get_properties(qids: set[str]) -> dict[str, list[str]]:
    """Return {qid: [pid, ...]} — the properties each Q-entity has claims for — using an on-disk cache."""
    cache = load_cache(PROPERTIES_CACHE_PATH)
    missing = sorted(q for q in qids if q not in cache)
    for i in range(0, len(missing), BATCH):
        chunk = missing[i : i + BATCH]
        cache.update(_fetch_properties(chunk))
        time.sleep(SLEEP)
        if i % (BATCH * 20) == 0:
            print(f"  fetched {i + len(chunk)}/{len(missing)} property sets")
    save_cache(PROPERTIES_CACHE_PATH, cache)
    return {q: cache.get(q, []) for q in qids}


def extract_topic_entities(query: str, labels: dict[str, str]) -> dict[str, str]:
    """Extract Q-entities referenced in the SPARQL query and map to their labels."""
    return {qid: labels.get(qid, "") for qid in QID_RE.findall(query)}


def extract_topic_relations(query: str, labels: dict[str, str]) -> dict[str, str]:
    """Extract P-properties referenced in the SPARQL query and map to their labels."""
    return {pid: labels.get(pid, "") for pid in PID_RE.findall(query)}


def extract_entity_relations(
    query: str,
    labels: dict[str, str],
    properties: dict[str, list[str]],
) -> dict[str, dict[str, str]]:
    """For each topic Q-entity in the query, return its Wikidata-linked properties mapped to labels."""
    return {
        qid: {pid: labels.get(pid, "") for pid in properties.get(qid, [])}
        for qid in QID_RE.findall(query)
    }


def fetch_query_metadata(queries: list[str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Fetch labels and Wikidata-linked properties for all Q/P-IDs in or associated with the queries.

    Returns (labels, properties):
      labels[id]       -> english label for every Q-ID and P-ID (from queries or properties)
      properties[qid]  -> list of P-IDs the topic entity has claims for on Wikidata
    """
    qids: set[str] = set()
    query_pids: set[str] = set()
    for q in queries:
        qids.update(QID_RE.findall(q))
        query_pids.update(PID_RE.findall(q))

    print(f"Fetching properties for {len(qids)} unique Q-IDs")
    properties = get_properties(qids)

    entity_pids: set[str] = set()
    for pids in properties.values():
        entity_pids.update(pids)

    all_ids = qids | query_pids | entity_pids
    print(f"Fetching labels for {len(all_ids)} unique Q/P-IDs")
    labels = get_labels(all_ids)

    return labels, properties
