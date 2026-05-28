"""
LLM-as-a-Judge: answer-level accuracy.

Asks an LLM to judge whether the predicted answer answers the question.
For each (pred, gold) pair, the judge sees, on each side (predicted and gold):
  - route, kb_id, and KB-specific schema/context,
  - the generated query (target_query on the gold side),
  - the answer materialized into readable form
    (SEARCH: top doc title+text from that KB; SQL/SPARQL/CYPHER: result values).
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.data.schema import RouteType, UnifiedSample
from src.evaluation.metrics import is_exact_match
from src.model.llm_client import LLMClient
from src.model.retrieval import (
    ExecutionResult, GeneratedQuery, RetrievalPipeline, RouteDecision,
)
from src.utils import execute_sql, execute_sparql, execute_cypher


JUDGE_SYSTEM = (
    "You are a strict but fair evaluator. You will see a user question and "
    "two sides: a PREDICTED side (the model's chosen KB) and a GOLD side "
    "(the labeled KB — known to be correct). Each side carries its KB "
    "schema/context, the query that was run, and the resulting answer.\n\n"
    "Decide whether the predicted side correctly answers the user question. "
    "There are two independent ways the prediction can be correct:\n\n"
    "1. ANSWER MATCH — the predicted answer is equivalent in meaning to the "
    "gold answer, allowing reordering, alias differences, formatting "
    "differences, or extra surrounding context.\n\n"
    "2. FAITHFUL IMPLEMENTATION ON A DIFFERENT KB — the predicted query "
    "faithfully realizes what the user asked, interpreted against the "
    "predicted KB's schema and data, and the predicted answer is what that "
    "query correctly produces. The values may differ entirely from gold "
    "because the predicted KB legitimately holds different content. This "
    "case applies whenever more than one knowledge base could reasonably "
    "answer the same kind of question.\n\n"
    "If the gold answer is empty or otherwise degenerate (the gold query "
    "may itself be buggy or stale), the gold reference is uninformative — "
    "judge by FAITHFUL IMPLEMENTATION alone in that case.\n\n"
    "Reject when the predicted answer is off-topic or when the predicted "
    "query plainly fails to capture what the question is asking. If both "
    "pred and gold answers are empty, ALWAYS count it as an ANSWER MATCH "
    "(this overrides any reasoning about whether the question should have a "
    "real-world answer). If only the predicted answer is empty, reject. "
    "Use the schemas and queries on both sides to interpret unfamiliar values."
)

JUDGE_INSTRUCTION = 'Respond with JSON: {"correct": true|false, "reason": "<one-line reason>"}'

JUDGE_SIDE_MAX_CHARS = 20000


def _execute_gold(gold: UnifiedSample, pipeline: RetrievalPipeline) -> ExecutionResult:
    """Run gold.target_query against the gold KB and wrap as an ExecutionResult for select_context."""
    if gold.route == RouteType.SEARCH:
        answer = gold.gold_answer
    elif gold.route == RouteType.SQL:
        assert gold.target_query is not None, f"target_query is None for '{gold.id}'"
        answer = execute_sql(gold.target_query, str(pipeline._find_db_path(gold.kb_id)))
    elif gold.route == RouteType.SPARQL:
        assert gold.target_query is not None, f"target_query is None for '{gold.id}'"
        answer = execute_sparql(gold.target_query)
    elif gold.route == RouteType.CYPHER:
        assert gold.target_query is not None, f"target_query is None for '{gold.id}'"
        answer = execute_cypher(gold.target_query, gold.kb_id)
    else:
        raise ValueError(f"Unsupported route: {gold.route}")
    return ExecutionResult(
        generated_query=GeneratedQuery(
            route_decision=RouteDecision(gold.question, gold.route, gold.kb_id),
            query=gold.target_query or "",
        ),
        answer=answer,
    )


def _format_side(label: str, exec_result: ExecutionResult, pipeline: RetrievalPipeline, metadata: dict | None) -> str:
    d = exec_result.generated_query.route_decision
    # Document Search query may hold the hypothetical passage; show the literal question instead (matches select()).
    query_line = d.question if d.route == RouteType.SEARCH else exec_result.generated_query.query
    side = (
        f"[{label}] route={d.route.value} | kb={d.kb_id}\n"
        f"query: {query_line}\n"
        f"{pipeline.select_context(exec_result, metadata)}"
    )
    return (
        side if len(side) <= JUDGE_SIDE_MAX_CHARS
        else side[:JUDGE_SIDE_MAX_CHARS] + "\n... [truncated]"
    )


def judge_one(
    pred: ExecutionResult,
    gold: UnifiedSample,
    llm: LLMClient,
    pipeline: RetrievalPipeline,
    pass_exact_match: bool = True,
) -> bool:
    gold_exec = _execute_gold(gold, pipeline)

    if pass_exact_match and is_exact_match(pred, gold_exec):
        return True

    metadata = gold.metadata or {}
    prompt = (
        f"Question: {gold.question}\n\n"
        f"{_format_side('PREDICTED', pred, pipeline, metadata)}\n\n"
        f"{_format_side('GOLD', gold_exec, pipeline, metadata)}\n\n"
        f"{JUDGE_INSTRUCTION}"
    )
    result = llm.generate_json(prompt, system=JUDGE_SYSTEM)
    return bool(result.get("correct", False))


def run_judge(
    pred: ExecutionResult,
    gold: UnifiedSample,
    llm: LLMClient,
    pipeline: RetrievalPipeline,
    pass_exact_match: bool = True,
) -> bool:
    try:
        return judge_one(pred, gold, llm, pipeline, pass_exact_match)
    except Exception as e:
        print(f"  judge for '{gold.id}' failed, counting as incorrect: {e}")
        return False


def evaluate_judge(
    predictions: list[ExecutionResult],
    ground_truths: list[UnifiedSample],
    llm: LLMClient,
    pipeline: RetrievalPipeline,
    pass_exact_match: bool = True,
    max_workers: int = 8,
) -> dict:
    """LLM-judged answer-level accuracy with per-gold-route breakdown."""
    assert len(predictions) == len(ground_truths), (
        f"Length mismatch: {len(predictions)} predictions vs {len(ground_truths)} ground truths"
    )

    n = len(predictions)
    if n == 0:
        return {"judge_accuracy": 0.0, "per_route_judge": {}, "n": 0}

    for pred, gold in zip(predictions, ground_truths):
        assert pred.generated_query.route_decision.question == gold.question, (
            f"Question mismatch: {pred.generated_query.route_decision.question!r} vs {gold.question!r}"
        )

    correct = 0
    per_route: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(run_judge, pred, gold, llm, pipeline, pass_exact_match): gold
            for pred, gold in zip(predictions, ground_truths)
        }
        for fut in tqdm(as_completed(futures), total=n, desc="Judging"):
            ok = fut.result()
            gold = futures[fut]
            correct += ok
            bucket = per_route[gold.route.value]
            bucket["total"] += 1
            bucket["correct"] += ok

    per_route_judge = {
        route_type: {"n": c["total"], "judge_accuracy": c["correct"] / c["total"]}
        for route_type, c in per_route.items()
    }
    return {"judge_accuracy": correct / n, "per_route_judge": per_route_judge, "n": n}
