"""Execute gold SPARQL queries from text-to-SPARQL datasets and cache the answers."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from src.data.datasets import load_samples
from src.utils import SPARQL_CACHE_PATHS, WIKIDATA_ENDPOINT, QLEVER_ENDPOINT, _sparql_caches, execute_sparql


SPARQL_DATASETS = ("lcquad2", "qald10", "simplequestions")


def load_cache(path: Path) -> dict[str, list[str]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_cache(path: Path, cache: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", choices=["qlever", "wdqs"], default="qlever")
    args = parser.parse_args()

    endpoint = QLEVER_ENDPOINT if args.endpoint == "qlever" else WIKIDATA_ENDPOINT
    cache_path = SPARQL_CACHE_PATHS[endpoint]

    samples = load_samples("data/processed", list(SPARQL_DATASETS), split="test")
    queries = {s.target_query for s in samples if s.target_query}

    cache = load_cache(cache_path)
    pending = [q for q in queries if not cache.get(q)]
    print(f"Endpoint: {endpoint}")
    print(f"To execute: {len(pending)}")

    # Without this, execute_sparql would return the cached empty result instead of re-executing the query.
    for q in pending:
        _sparql_caches[endpoint].pop(q, None)

    for q in tqdm(pending):
        cache[q] = execute_sparql(q, endpoint=endpoint)
        time.sleep(0.5)

    save_cache(cache_path, cache)
    print(f"Saved {len(cache)} entries to {cache_path}")


if __name__ == "__main__":
    main()
