#!/usr/bin/env python3
"""Evaluate a saved run, or evaluate results in-memory."""

import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.data.schema import RouteType, UnifiedSample
from src.model.llm_client import LLMClient
from src.model.retrieval import CandidateResults, ExecutionResult, RetrievalPipeline
from src.evaluation.metrics import evaluate_routing, evaluate_generation, evaluate_search, evaluate_sql, evaluate_sparql, evaluate_cypher, evaluate_selector
from src.evaluation.judge import evaluate_judge
from src.utils import load_run, save_run


def _macro(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _evaluate_predictions(samples: list[UnifiedSample], results: list[ExecutionResult]) -> dict:
    # Evaluate source selection
    print(f"\n{'=' * 60}")
    print("Routing Evaluation")
    print(f"{'=' * 60}")
    routing_metrics = evaluate_routing([r.generated_query.route_decision for r in results], samples)
    routing_breakdowns = list(routing_metrics["per_route_accuracy"].values())
    routing_metrics["macro_route_accuracy"] = _macro([b["route_accuracy"] for b in routing_breakdowns])
    routing_metrics["macro_kb_accuracy"] = _macro([b["kb_accuracy"] for b in routing_breakdowns])
    routing_metrics["macro_overall_accuracy"] = _macro([b["overall_accuracy"] for b in routing_breakdowns])
    print(f"Micro average:    "
          f"route={routing_metrics['route_accuracy']:.2%}  "
          f"kb={routing_metrics['kb_accuracy']:.2%}  "
          f"overall={routing_metrics['overall_accuracy']:.2%}")
    print(f"Macro average:    "
          f"route={routing_metrics['macro_route_accuracy']:.2%}  "
          f"kb={routing_metrics['macro_kb_accuracy']:.2%}  "
          f"overall={routing_metrics['macro_overall_accuracy']:.2%}")
    for route_type, breakdown in routing_metrics["per_route_accuracy"].items():
        print(f"  {route_type:8s}  n={breakdown['n']}  "
              f"route={breakdown['route_accuracy']:.2%}  "
              f"kb={breakdown['kb_accuracy']:.2%}  "
              f"overall={breakdown['overall_accuracy']:.2%}")

    # Evaluate query formulation
    print(f"\n{'=' * 60}")
    print("Generation Evaluation")
    print(f"{'=' * 60}")
    generation_metrics = evaluate_generation([r.generated_query for r in results], samples)
    generation_breakdowns = list(generation_metrics["per_route_generation"].values())
    generation_metrics["macro_exact_match"] = _macro([b["exact_match"] for b in generation_breakdowns])
    generation_metrics["macro_f1"] = _macro([b["f1"] for b in generation_breakdowns])
    print(f"Micro average:    "
          f"exact_match={generation_metrics['exact_match']:.2%}  "
          f"f1={generation_metrics['f1']:.2%}")
    print(f"Macro average:    "
          f"exact_match={generation_metrics['macro_exact_match']:.2%}  "
          f"f1={generation_metrics['macro_f1']:.2%}")
    for route_type, breakdown in generation_metrics["per_route_generation"].items():
        print(f"  {route_type:8s}  n={breakdown['n']}  "
              f"exact_match={breakdown['exact_match']:.2%}  "
              f"f1={breakdown['f1']:.2%}")

    # Evaluate execution (grouped by gold paradigm)
    print(f"\n{'=' * 60}")
    print("Execution Evaluation")
    print(f"{'=' * 60}")
    by_route = {rt: ([], []) for rt in RouteType}
    for r, s in zip(results, samples):
        assert r.generated_query.route_decision.question == s.question, (
            f"Question mismatch: {r.generated_query.route_decision.question!r} vs {s.question!r}"
        )
        by_route[s.route][0].append(r)
        by_route[s.route][1].append(s)

    search_metrics = evaluate_search(*by_route[RouteType.SEARCH])
    sql_metrics = evaluate_sql(*by_route[RouteType.SQL])
    sparql_metrics = evaluate_sparql(*by_route[RouteType.SPARQL])
    cypher_metrics = evaluate_cypher(*by_route[RouteType.CYPHER])

    per_route_scores = [
        ("SEARCH", search_metrics, search_metrics.get("NDCG@10", 0)),
        ("SQL", sql_metrics, sql_metrics.get("execution_accuracy", 0)),
        ("SPARQL", sparql_metrics, sparql_metrics.get("execution_accuracy", 0)),
        ("CYPHER", cypher_metrics, cypher_metrics.get("execution_accuracy", 0)),
    ]
    total_n = sum(m["n"] for _, m, _ in per_route_scores)
    micro_average = sum(m["n"] * s for _, m, s in per_route_scores) / total_n if total_n else 0.0
    macro_average = _macro([s for _, m, s in per_route_scores if m["n"] > 0])
    print(f"Micro average:    {micro_average:.2%}  (NDCG@10 for SEARCH, execution_accuracy for SQL/SPARQL/CYPHER)")
    print(f"Macro average:    {macro_average:.2%}  (NDCG@10 for SEARCH, execution_accuracy for SQL/SPARQL/CYPHER)")
    print(f"SEARCH  n={search_metrics['n']}  NDCG@10={search_metrics.get('NDCG@10', 0):.4f}  Recall@10={search_metrics.get('Recall@10', 0):.4f}")
    print(f"SQL     n={sql_metrics['n']}  execution_accuracy={sql_metrics['execution_accuracy']:.2%}")
    print(f"SPARQL  n={sparql_metrics['n']}  execution_accuracy={sparql_metrics['execution_accuracy']:.2%}")
    print(f"CYPHER  n={cypher_metrics['n']}  execution_accuracy={cypher_metrics['execution_accuracy']:.2%}")

    return {
        "routing": routing_metrics,
        "generation": generation_metrics,
        "execution": {
            "micro_average": micro_average,
            "macro_average": macro_average,
            "SEARCH": search_metrics,
            "SQL": sql_metrics,
            "SPARQL": sparql_metrics,
            "CYPHER": cypher_metrics,
        },
    }


def _pick_oracle(sample: UnifiedSample, cr: CandidateResults) -> ExecutionResult:
    """Pick the candidate whose (route, kb_id) matches gold; falls back to the first candidate."""
    for c in cr.candidates:
        if (
            c.generated_query.route_decision.route == sample.route
            and c.generated_query.route_decision.kb_id == sample.kb_id
        ):
            return c
    return cr.candidates[0]


def _evaluate_judge(
    samples: list[UnifiedSample],
    predictions: list[ExecutionResult],
    llm: LLMClient,
    pipeline: RetrievalPipeline,
) -> dict:
    print(f"\n{'=' * 60}\nJudge Evaluation\n{'=' * 60}")
    judge_metrics = evaluate_judge(predictions, samples, llm, pipeline)
    judge_metrics["macro_judge_accuracy"] = _macro([b["judge_accuracy"] for b in judge_metrics["per_route_judge"].values()])
    print(f"Micro average:    judge_accuracy={judge_metrics['judge_accuracy']:.2%}  (n={judge_metrics['n']})")
    print(f"Macro average:    judge_accuracy={judge_metrics['macro_judge_accuracy']:.2%}")
    for route_type, breakdown in judge_metrics["per_route_judge"].items():
        print(f"  {route_type:8s}  n={breakdown['n']}  judge_accuracy={breakdown['judge_accuracy']:.2%}")
    return judge_metrics


def evaluate(
    samples: list[UnifiedSample],
    results: list[CandidateResults],
    judge_llm: LLMClient | None = None,
    pipeline: RetrievalPipeline | None = None,
) -> dict:
    print(f"\n{'#' * 60}\n# Top-1 (most-confident candidate)\n{'#' * 60}")
    top1_picks = [r.candidates[0] for r in results]
    top1_metrics = _evaluate_predictions(samples, top1_picks)
    top1_metrics["judge"] = _evaluate_judge(samples, top1_picks, judge_llm, pipeline) if judge_llm and pipeline else None

    selected_metrics = oracle_metrics = selector_metrics = None
    if not all(len(r.candidates) == 1 for r in results):
        print(f"\n{'#' * 60}\n# Selected predictions\n{'#' * 60}")
        selected_picks = [r.best for r in results]
        selected_metrics = _evaluate_predictions(samples, selected_picks)
        selected_metrics["judge"] = _evaluate_judge(samples, selected_picks, judge_llm, pipeline) if judge_llm and pipeline else None

        print(f"\n{'#' * 60}\n# Oracle (per-sample candidate matching gold route + kb_id)\n{'#' * 60}")
        oracle_picks = [_pick_oracle(s, cr) for s, cr in zip(samples, results)]
        oracle_metrics = _evaluate_predictions(samples, oracle_picks)
        oracle_metrics["judge"] = _evaluate_judge(samples, oracle_picks, judge_llm, pipeline) if judge_llm and pipeline else None

        print(f"\n{'#' * 60}\n# Selector Accuracy\n{'#' * 60}")
        selector_metrics = evaluate_selector(results, samples)
        print(f"n={selector_metrics['n']}  accuracy={selector_metrics['accuracy']:.2%}")

    return {
        "top1": top1_metrics,
        "selected": selected_metrics,
        "all_candidates": oracle_metrics,
        "selector": selector_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved run.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to a saved run directory.")
    parser.add_argument("--judge", action="store_true", help="Run the LLM-as-a-Judge metric on selected predictions.")
    parser.add_argument("--judge-provider", type=str, default="openai", choices=["openai", "anthropic", "google", "vllm"])
    parser.add_argument("--judge-model", type=str, default=None)
    cli_args = parser.parse_args()
    run_dir = Path(cli_args.run_dir)

    args, samples, results = load_run(run_dir)
    judge_llm = LLMClient(provider=cli_args.judge_provider, model=cli_args.judge_model) if cli_args.judge else None
    pipeline = RetrievalPipeline(llm=judge_llm, data_dir="data/processed") if judge_llm else None
    metrics = evaluate(samples, results, judge_llm=judge_llm, pipeline=pipeline)
    save_run(run_dir, args, samples, results, metrics)


if __name__ == "__main__":
    main()
