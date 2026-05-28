"""Execute gold Cypher queries from text2cypher and cache the answers."""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from src.data.datasets import load_samples
from src.utils import CYPHER_CACHE_PATH, _cypher_cache, execute_cypher


CYPHER_DATASETS = ("text2cypher",)


def load_cache() -> dict[str, dict[str, list[str]]]:
    return (
        json.loads(CYPHER_CACHE_PATH.read_text(encoding="utf-8"))
        if CYPHER_CACHE_PATH.exists() else {}
    )


def save_cache(cache: dict[str, dict[str, list[str]]]) -> None:
    CYPHER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CYPHER_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    samples = load_samples("data/processed", list(CYPHER_DATASETS), split="test")
    pairs = {(s.kb_id, s.target_query) for s in samples if s.target_query}

    cache = load_cache()
    pending = [(kb, q) for kb, q in pairs if not cache.get(kb, {}).get(q)]
    print(f"To execute: {len(pending)}")

    # Without this, execute_cypher would return the cached empty result instead of re-executing.
    for kb, q in pending:
        if kb in _cypher_cache:
            _cypher_cache[kb].pop(q, None)

    for kb, q in tqdm(pending):
        cache.setdefault(kb, {})[q] = execute_cypher(q, kb)
        time.sleep(0.5)

    save_cache(cache)
    print(f"Saved {sum(len(v) for v in cache.values())} entries to {CYPHER_CACHE_PATH}")


if __name__ == "__main__":
    main()
    os._exit(0)  # force-exit past any stuck Neo4j worker threads
