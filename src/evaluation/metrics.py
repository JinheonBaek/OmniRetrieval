"""
Evaluation metrics for the OmniRetrieval pipeline.

1. Source selection    — does it pick the right (route_type, kb_id)?
2. Query formulation   — does the formulated query match the target query?
3. Retrieval quality   — NDCG, MAP, Recall, Precision for Document Search.
4. Execution Match     — do predicted results match gold for SQL/SPARQL/Cypher?
"""

from __future__ import annotations

import ast
import collections
import re
import string

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.data.schema import RouteType, UnifiedSample
from src.model.retrieval import RouteDecision, GeneratedQuery, ExecutionResult, CandidateResults
from src.utils import execute_sql, execute_sparql, execute_cypher


# neo4j.time.* values can't be parsed by ast.literal_eval; rewrite to quoted strings first.
NEO4J_TIME_RE = re.compile(r"neo4j\.time\.\w+\([^)]*\)")


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def evaluate_routing(
    predictions: list[RouteDecision],
    ground_truths: list[UnifiedSample],
) -> dict:
    """
    Measure source-selection accuracy: backend and KB identification.

    Args:
        predictions:   list of RouteDecision (predicted route + kb_id).
        ground_truths: list of UnifiedSample (gold route + kb_id), same order.

    Returns:
        Dict with overall accuracy, route-only accuracy, kb-only accuracy,
        and per-route breakdown.
    """
    assert len(predictions) == len(ground_truths), (
        f"Length mismatch: {len(predictions)} predictions vs {len(ground_truths)} ground truths"
    )

    n = len(predictions)
    if n == 0:
        return {"overall_accuracy": 0.0, "route_accuracy": 0.0, "kb_accuracy": 0.0, "per_route": {}, "n": 0}

    route_correct = 0
    kb_correct = 0
    both_correct = 0

    # per-route counters: route_type -> {total, route_correct, kb_correct, both_correct}
    per_route: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "route_correct": 0, "kb_correct": 0, "both_correct": 0})

    for pred, gold in zip(predictions, ground_truths):
        assert pred.question == gold.question, (
            f"Question mismatch: {pred.question!r} vs {gold.question!r}"
        )
        gold_route = gold.route.value
        pred_route = pred.route.value

        r_match = pred_route == gold_route
        k_match = pred.kb_id == gold.kb_id
        b_match = r_match and k_match

        route_correct += r_match
        kb_correct += k_match
        both_correct += b_match

        bucket = per_route[gold_route]
        bucket["total"] += 1
        bucket["route_correct"] += r_match
        bucket["kb_correct"] += k_match
        bucket["both_correct"] += b_match

    per_route_accuracy = {
        route_type: {
            "n": counts["total"],
            "route_accuracy": counts["route_correct"] / counts["total"],
            "kb_accuracy": counts["kb_correct"] / counts["total"],
            "overall_accuracy": counts["both_correct"] / counts["total"],
        }
        for route_type, counts in per_route.items()
    }

    return {
        "overall_accuracy": both_correct / n,
        "route_accuracy": route_correct / n,
        "kb_accuracy": kb_correct / n,
        "per_route_accuracy": per_route_accuracy,
        "n": n,
    }


def evaluate_generation(
    predictions: list[GeneratedQuery],
    ground_truths: list[UnifiedSample],
) -> dict:
    """
    Measure query-formulation quality: exact match and token-level F1
    between the formulated and target query.

    Args:
        predictions:   list of GeneratedQuery (predicted query string).
        ground_truths: list of UnifiedSample (gold target_query), same order.

    Returns:
        Dict with overall exact-match and F1 scores, and per-route breakdown.
    """
    assert len(predictions) == len(ground_truths), (
        f"Length mismatch: {len(predictions)} predictions vs {len(ground_truths)} ground truths"
    )

    n = 0
    em_sum = 0
    f1_sum = 0.0

    # per-route counters: route_type -> {total, em, f1}
    per_route: dict[str, dict[str, float]] = defaultdict(lambda: {"total": 0, "em": 0, "f1": 0.0})

    for pred, gold in zip(predictions, ground_truths):
        assert pred.route_decision.question == gold.question, (
            f"Question mismatch: {pred.route_decision.question!r} vs {gold.question!r}"
        )
        if gold.target_query is None:
            continue

        em = compute_exact(gold.target_query, pred.query)
        f1 = compute_f1(gold.target_query, pred.query)
        n += 1
        em_sum += em
        f1_sum += f1

        bucket = per_route[gold.route.value]
        bucket["total"] += 1
        bucket["em"] += em
        bucket["f1"] += f1

    if n == 0:
        return {"exact_match": 0.0, "f1": 0.0, "per_route_generation": {}, "n": 0}

    per_route_generation = {
        route_type: {
            "n": counts["total"],
            "exact_match": counts["em"] / counts["total"],
            "f1": counts["f1"] / counts["total"],
        }
        for route_type, counts in per_route.items()
    }

    return {
        "exact_match": em_sum / n,
        "f1": f1_sum / n,
        "per_route_generation": per_route_generation,
        "n": n,
    }


def _sorted_cells(row_str: str) -> tuple:
    # Parse a stringified row (e.g. "(24.2, 'Chevrolet')") and return its cells
    # as a sorted tuple, so column-order differences between gold and predicted
    # SQL do not produce false-negative comparisons.
    try:
        cells = ast.literal_eval(row_str)
    except (ValueError, SyntaxError):
        return (row_str,)
    if not isinstance(cells, tuple):
        cells = (cells,)
    return tuple(sorted(map(repr, cells)))


def _cypher_atoms(rows: list[str]) -> set[str]:
    # Flatten Cypher rows into a set of atomic stringified values, unpacking dict/list cells.
    atoms: set[str] = set()
    def walk(v):
        if isinstance(v, dict):
            for x in v.values(): walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v: walk(x)
        else:
            atoms.add(repr(v))
    for r in rows:
        try:
            walk(ast.literal_eval(NEO4J_TIME_RE.sub(lambda m: repr(m.group()), r)))
        except (ValueError, SyntaxError):
            atoms.add(r)
    return atoms


def _cypher_rows_match(pred_rows: list[str], gold_rows: list[str]) -> bool:
    """
    Symmetric match: same row count, and the atomic values of one side are a subset of the other.

    Accepts both 'gold projects narrower than pred' (gold `RETURN m.title`, pred `RETURN m`)
    and 'pred projects narrower than gold' (pred drops a sort key or filter echo gold kept).
    """
    return (
        len(pred_rows) == len(gold_rows)
        and (
            _cypher_atoms(gold_rows) <= _cypher_atoms(pred_rows)
            or _cypher_atoms(pred_rows) <= _cypher_atoms(gold_rows)
        )
    )


def is_exact_match(pred: ExecutionResult, gold: ExecutionResult) -> bool:
    """Per-route binary equality between pred and gold execution results."""
    if pred.generated_query.route_decision.route != gold.generated_query.route_decision.route:
        return False
    route = gold.generated_query.route_decision.route
    if route == RouteType.SEARCH:
        return all(doc_id in pred.answer for doc_id in gold.answer)
    if route == RouteType.SQL:
        return sorted(map(_sorted_cells, pred.answer)) == sorted(map(_sorted_cells, gold.answer))
    if route == RouteType.CYPHER:
        return _cypher_rows_match(pred.answer, gold.answer)
    if route == RouteType.SPARQL:
        return sorted(pred.answer) == sorted(gold.answer)
    raise ValueError(f"Unsupported route: {route}")


def evaluate_search(
    predictions: list[ExecutionResult],
    ground_truths: list[UnifiedSample],
    k_values: list[int] = [1, 3, 5, 10, 100],
) -> dict:
    """
    Measure retrieval quality using standard IR metrics (NDCG, MAP, Recall, Precision).

    Uses pytrec_eval via BEIR (EvaluateRetrieval.evaluate).

    Args:
        predictions:   list of ExecutionResult with scores dict.
        ground_truths: list of UnifiedSample with gold_answer doc IDs.
        k_values:      cutoff values for the metrics.

    Returns:
        Dict with NDCG@k, MAP@k, Recall@k, P@k for each k.
    """
    assert len(predictions) == len(ground_truths)

    n = len(predictions)
    if n == 0:
        return {"n": 0}

    from beir.retrieval.evaluation import EvaluateRetrieval

    # Build qrels and results in pytrec_eval format
    qrels: dict[str, dict[str, int]] = {}
    results: dict[str, dict[str, float]] = {}

    for i, (pred, gold) in enumerate(zip(predictions, ground_truths)):
        assert pred.generated_query.route_decision.question == gold.question, (
            f"Question mismatch at index {i}: "
            f"pred={pred.generated_query.route_decision.question!r}, gold={gold.question!r}"
        )
        qid = gold.id or str(i)
        qrels[qid] = {doc_id: 1 for doc_id in gold.gold_answer}
        results[qid] = pred.scores

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results, k_values, ignore_identical_ids=False,
    )

    return {**ndcg, **_map, **recall, **precision, "n": n}


def _check_sql(pred: ExecutionResult, gold: UnifiedSample, data_dir: str) -> bool:
    assert gold.target_query is not None, f"target_query is None for '{gold.id}'"
    db_path = Path(data_dir) / gold.source / "databases" / gold.kb_id / f"{gold.kb_id}.sqlite"
    gold_answer = execute_sql(gold.target_query, str(db_path))
    return sorted(map(_sorted_cells, pred.answer)) == sorted(map(_sorted_cells, gold_answer))


def evaluate_sql(
    predictions: list[ExecutionResult],
    ground_truths: list[UnifiedSample],
    data_dir: str = "data/processed",
    max_workers: int = 8,
) -> dict:
    """
    Measure execution accuracy for SQL queries.

    The gold query (target_query) is always executed against the database
    to obtain the expected results.
    """
    assert len(predictions) == len(ground_truths)

    n = len(predictions)
    if n == 0:
        return {"execution_accuracy": 0.0, "n": 0}

    for pred, gold in zip(predictions, ground_truths):
        assert pred.generated_query.route_decision.question == gold.question, (
            f"Question mismatch: "
            f"pred={pred.generated_query.route_decision.question!r}, gold={gold.question!r}"
        )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_check_sql, p, g, data_dir) for p, g in zip(predictions, ground_truths)]
        correct = sum(f.result() for f in futures)

    return {"execution_accuracy": correct / n, "n": n}


def _check_sparql(pred: ExecutionResult, gold: UnifiedSample) -> bool:
    assert gold.target_query is not None, f"target_query is None for '{gold.id}'"
    gold_answer = execute_sparql(gold.target_query)
    return sorted(pred.answer) == sorted(gold_answer)


def evaluate_sparql(
    predictions: list[ExecutionResult],
    ground_truths: list[UnifiedSample],
    max_workers: int = 8,
) -> dict:
    """
    Measure execution accuracy for SPARQL queries.

    The gold query (target_query) is always executed against the Wikidata
    endpoint to obtain the expected results.
    """
    assert len(predictions) == len(ground_truths)

    n = len(predictions)
    if n == 0:
        return {"execution_accuracy": 0.0, "n": 0}

    for pred, gold in zip(predictions, ground_truths):
        assert pred.generated_query.route_decision.question == gold.question, (
            f"Question mismatch: "
            f"pred={pred.generated_query.route_decision.question!r}, gold={gold.question!r}"
        )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_check_sparql, p, g) for p, g in zip(predictions, ground_truths)]
        correct = sum(f.result() for f in futures)

    return {"execution_accuracy": correct / n, "n": n}


def _check_cypher(pred: ExecutionResult, gold: UnifiedSample) -> bool:
    assert gold.target_query is not None, f"target_query is None for '{gold.id}'"
    gold_answer = execute_cypher(gold.target_query, gold.kb_id)
    return _cypher_rows_match(pred.answer, gold_answer)


def evaluate_cypher(
    predictions: list[ExecutionResult],
    ground_truths: list[UnifiedSample],
    max_workers: int = 8,
) -> dict:
    """
    Measure execution accuracy for CYPHER queries.

    The gold query (target_query) is always executed against the Neo4j
    Labs demo sandbox (typically a cache hit) to obtain the expected results.
    """
    assert len(predictions) == len(ground_truths)

    n = len(predictions)
    if n == 0:
        return {"execution_accuracy": 0.0, "n": 0}

    for pred, gold in zip(predictions, ground_truths):
        assert pred.generated_query.route_decision.question == gold.question, (
            f"Question mismatch: "
            f"pred={pred.generated_query.route_decision.question!r}, gold={gold.question!r}"
        )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_check_cypher, p, g) for p, g in zip(predictions, ground_truths)]
        correct = sum(f.result() for f in futures)

    return {"execution_accuracy": correct / n, "n": n}


def evaluate_selector(
    candidate_results: list[CandidateResults],
    ground_truths: list[UnifiedSample],
) -> dict:
    """Selector accuracy on samples where >=1 candidate matches gold (route, kb_id)."""
    assert len(candidate_results) == len(ground_truths)

    n = len(candidate_results)
    if n == 0:
        return {"accuracy": 0.0, "n": 0}

    n_match_exists = 0
    correct = 0
    for cr, gold in zip(candidate_results, ground_truths):
        cand_routes = [
            (c.generated_query.route_decision.route, c.generated_query.route_decision.kb_id)
            for c in cr.candidates
        ]
        if (gold.route, gold.kb_id) not in cand_routes:
            continue
        n_match_exists += 1
        if cand_routes[cr.selected] == (gold.route, gold.kb_id):
            correct += 1

    return {"accuracy": correct / n_match_exists if n_match_exists else 0.0, "n": n_match_exists}
