"""Shared utility functions."""

import concurrent.futures
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, cast

import requests
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import ClientError
from neo4j.graph import Node, Relationship
from SPARQLWrapper import SPARQLWrapper, JSON
from SPARQLWrapper.SPARQLExceptions import QueryBadFormed, EndPointInternalError

from src.data.cypher_corpora import NEO4J_URI, ALIAS_TO_DB
from src.data.schema import UnifiedSample, load_jsonl, save_jsonl

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
QLEVER_ENDPOINT = "https://qlever.dev/api/wikidata"
WIKIDATA_ENTITY_PREFIX = "http://www.wikidata.org/entity/"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_QID_RE = re.compile(r"^Q\d+$")
WIKIDATA_BATCH = 50
WIKIDATA_SLEEP = 0.2

# QLever has no implicit prefixes (WDQS does). Prepended to every query in execute_sparql.
# Covers every prefix used by ground-truth SPARQL in data/processed/{lcquad2,qald10,simplequestions}.
WIKIDATA_PREFIXES = """\
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX psn: <http://www.wikidata.org/prop/statement/value-normalized/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX dct: <http://purl.org/dc/terms/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""

SPARQL_CACHE_PATHS: dict[str, Path] = {
    WIKIDATA_ENDPOINT: Path("data/cache/wikidata/sparqls_wdqs.json"),
    QLEVER_ENDPOINT: Path("data/cache/wikidata/sparqls_qlever.json"),
}
_sparql_caches: dict[str, dict[str, list[str]]] = {
    endpoint: (json.loads(path.read_text(encoding="utf-8")) if path.exists() else {})
    for endpoint, path in SPARQL_CACHE_PATHS.items()
}

WIKIDATA_LABELS_CACHE_PATH = Path("data/cache/wikidata/labels.json")
_wikidata_labels: dict[str, str] = (
    json.loads(WIKIDATA_LABELS_CACHE_PATH.read_text(encoding="utf-8"))
    if WIKIDATA_LABELS_CACHE_PATH.exists() else {}
)

CYPHER_CACHE_PATH = Path("data/cache/neo4j/cyphers.json")
_cypher_cache: dict[str, dict[str, list[str]]] = (
    json.loads(CYPHER_CACHE_PATH.read_text(encoding="utf-8"))
    if CYPHER_CACHE_PATH.exists() else {}
)
_neo4j_drivers: dict[str, Driver] = {}


def wikidata_api_request(params: dict) -> dict:
    """GET the Wikidata API with retry-on-error and Retry-After honoring."""
    for attempt in range(5):
        try:
            r = requests.get(WIKIDATA_API_URL, params=params, timeout=30,
                             headers={"User-Agent": "OmniRetrieval/1.0 (https://github.com/JinheonBaek/OmniRetrieval)"})
            r.raise_for_status()
            return r.json().get("entities", {})
        except (requests.RequestException, ValueError) as e:
            response = getattr(e, "response", None)
            retry_after = response.headers.get("Retry-After", "") if response is not None else ""
            wait = int(retry_after) if retry_after.isdigit() else 2 ** attempt
            print(f"  retry {attempt + 1}/5 after {wait}s: {e}")
            time.sleep(wait)
    raise RuntimeError("Wikidata API failed after 5 retries")


def fetch_wikidata_labels(ids: list[str]) -> dict[str, str]:
    """Fetch English labels for Wikidata QIDs (returns empty string for missing labels)."""
    entities = wikidata_api_request({
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


def hydrate_wikidata_qids(answer: list[str]) -> list[str]:
    """Append English labels to Wikidata QIDs in answer (uses in-memory cache; fetches misses)."""
    missing = sorted(a for a in answer if WIKIDATA_QID_RE.fullmatch(a) and a not in _wikidata_labels)
    for i in range(0, len(missing), WIKIDATA_BATCH):
        chunk = missing[i : i + WIKIDATA_BATCH]
        _wikidata_labels.update(fetch_wikidata_labels(chunk))
        time.sleep(WIKIDATA_SLEEP)
    return [f"{a} ({_wikidata_labels[a]})" if _wikidata_labels.get(a) else a for a in answer]


def execute_sql(query: str, db_path: str, timeout: float = 300.0, max_rows: int = 3000000) -> list[str]:
    """Execute a SQL query and return results as stringified rows (capped at max_rows)."""
    conn = sqlite3.connect(db_path)
    conn.set_progress_handler((lambda d=time.monotonic() + timeout: 1 if time.monotonic() > d else 0), 1000)
    try:
        rows = conn.execute(query).fetchmany(max_rows)
        return [str(row) for row in rows]
    except Exception:
        return []
    finally:
        conn.close()


def _get_neo4j_driver(db: str) -> Driver:
    """Return a cached, long-lived Neo4j driver for one demo DB (one auth per DB)."""
    if db not in _neo4j_drivers:
        _neo4j_drivers[db] = GraphDatabase.driver(
            NEO4J_URI, auth=(db, db),
            notifications_min_severity="OFF",  # silence DEPRECATION/UNRECOGNIZED warnings on stdout
        )
    return _neo4j_drivers[db]


def _normalize_cypher_value(v: Any) -> Any:
    """Strip Neo4j-internal element_id from Node/Relationship values so equal content compares equal."""
    if isinstance(v, (Node, Relationship)):
        return dict(v)
    if isinstance(v, list):
        return [_normalize_cypher_value(x) for x in v]
    return v


def execute_cypher(query: str, kb_id: str, timeout: float = 60.0) -> list[str]:
    """Execute a Cypher query against a neo4jlabs demo DB and return stringified rows."""
    if query in _cypher_cache.get(kb_id, {}):
        return _cypher_cache[kb_id][query]

    db = ALIAS_TO_DB.get(kb_id)
    if db is None:
        return []

    def _run() -> list[str]:
        with _get_neo4j_driver(db).session(database=db) as session:
            result = session.run(cast(Any, query), timeout=timeout)
            return [str(tuple(_normalize_cypher_value(v) for v in r.values())) for r in result]

    for attempt in range(5):
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            return ex.submit(_run).result(timeout=timeout)
        except ClientError:
            return []
        except Exception as e:
            wait = 2 ** attempt
            print(f"  retry {attempt + 1}/5 after {wait}s: {str(e)[:200]}")
            time.sleep(wait)
        finally:
            ex.shutdown(wait=False)
    return []


def execute_sparql(query: str, endpoint: str = QLEVER_ENDPOINT, timeout: float = 60.0) -> list[str]:
    """Execute a SPARQL query against a SPARQL endpoint.

    Parses results following the same convention as the QALD-10 preprocessor:
    - URI answers: extract Wikidata entity ID (e.g. Q2212)
    - Literal answers: keep the value as-is
    - Boolean answers (ASK queries): "true" or "false"
    """
    if query in _sparql_caches.get(endpoint, {}):
        return _sparql_caches[endpoint][query]

    # The strict contains() in QLever rejects numeric YEAR()/MONTH()/DAY() args; wrap in STR() to fix lcquad2 gold queries.
    query = re.sub(r'contains\s*\(\s*((?:YEAR|MONTH|DAY)\s*\(\s*\?\w+\s*\))', r'contains(STR(\1)', query, flags=re.IGNORECASE)
    sparql = SPARQLWrapper(endpoint, agent="OmniRetrieval/1.0 (https://github.com/JinheonBaek/OmniRetrieval)")
    sparql.setQuery(WIKIDATA_PREFIXES + query)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(int(timeout))

    def _run() -> dict[str, Any]:
        return cast(dict[str, Any], sparql.query().convert())

    for attempt in range(5):
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            results = ex.submit(_run).result(timeout=timeout)
            break
        except (QueryBadFormed, EndPointInternalError):
            return []
        except Exception as e:
            headers = getattr(e, "headers", None)
            retry_after = headers.get("Retry-After", "") if headers is not None else ""
            wait = int(retry_after) if retry_after.isdigit() else 2 ** attempt
            print(f"  retry {attempt + 1}/5 after {wait}s: {str(e)[:200]}")
            time.sleep(wait)
        finally:
            ex.shutdown(wait=False)
    else:
        return []

    # Boolean answers (ASK queries)
    if "boolean" in results:
        return [str(results["boolean"]).lower()]

    # Binding-based answers (SELECT queries)
    answer = []
    for binding in results.get("results", {}).get("bindings", []):
        for cell in binding.values():
            content = cell["value"]
            if cell["type"] == "uri" and content.startswith(WIKIDATA_ENTITY_PREFIX):
                content = content[len(WIKIDATA_ENTITY_PREFIX):]
            answer.append(content)
    return answer


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[UnifiedSample], list]:
    """Load run args, samples, and candidate results from a run directory."""
    from src.model.retrieval import CandidateResults
    args = json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
    samples = load_jsonl(str(run_dir / "samples.jsonl"))
    results = []
    with open(run_dir / "results.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            results.append(CandidateResults.from_dict(json.loads(line)))
    print(f"Loaded {len(results)} results from {run_dir}")
    return args, samples, results


def save_run(run_dir: Path, args: dict[str, Any], samples: list[UnifiedSample], results: list, metrics: dict) -> None:
    """Save run args, samples, candidate results, and evaluation metrics to a run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(args, indent=2), encoding="utf-8")
    save_jsonl(samples, str(run_dir / "samples.jsonl"))
    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} results and metrics to {run_dir}")
