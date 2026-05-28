#!/usr/bin/env python3
"""Run the OmniRetrieval pipeline."""

import argparse
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import cast

from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

from src.data.datasets import DATASET_PATHS, DEMO_QUERIES, load_samples, balance_by_route
from src.data.schema import RouteType
from src.model.llm_client import LLMClient
from src.model.retrieval import (
    RouteDecision, GeneratedQuery, ExecutionResult, CandidateResults, RetrievalPipeline,
)
from src.utils import save_run
from evaluate import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OmniRetrieval pipeline")
    parser.add_argument(
        "--demo", action="store_true",
        help="Use demo queries instead of processed datasets.",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        choices=list(DATASET_PATHS.keys()),
        help="Datasets to evaluate on (default: all).",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples to load per dataset.",
    )
    parser.add_argument(
        "--max-per-route", type=int, default=None,
        help="Max samples per route type for balancing (default: minority size).",
    )
    parser.add_argument(
        "--balance", action="store_true",
        help="Enable route balancing (off by default).",
    )
    parser.add_argument(
        "--no-metadata", action="store_true",
        help="Ignore sample metadata when building the per-sample context (ablation).",
    )
    parser.add_argument(
        "--gold-route", action="store_true",
        help="Skip routing and use the gold (route, kb_id) from each sample.",
    )
    parser.add_argument(
        "--single-source", type=str, default=None,
        choices=[r.value for r in RouteType],
        help="Restrict routing to a single backend (single-source baseline).",
    )
    parser.add_argument(
        "--top-k-routes", type=int, default=1,
        help="Number of routing decisions to consider per query (default 1).",
    )
    parser.add_argument(
        "--runs-dir", type=str, default="runs",
        help="Root directory for saved runs (default: runs/).",
    )
    parser.add_argument(
        "--provider", type=str, default="openai",
        choices=["openai", "anthropic", "google", "vllm"],
        help="LLM provider (default: openai).",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name (default: provider's default).",
    )
    parser.add_argument(
        "--enable-thinking", action="store_true",
        help="Enable thinking mode for vllm chat templates that support it (e.g. Qwen3.5).",
    )
    parser.add_argument(
        "--judge", action="store_true",
        help="Run the LLM-as-a-Judge metric on selected predictions (uses OpenAI gpt-5.4-mini).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Number of concurrent worker threads for the per-sample loop. "
             "Only applied for API providers (openai/anthropic/google); vllm stays serial.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def process_sample(sample, retriever, args) -> CandidateResults:
    sample_metadata = None if args.no_metadata else sample.metadata
    decisions = (
        [RouteDecision(sample.question, sample.route, sample.kb_id)]
        if args.gold_route else retriever.route(
            sample.question,
            top_k=args.top_k_routes,
            single_source=RouteType(args.single_source) if args.single_source else None,
        )
    )
    candidates = []
    for decision in decisions:
        generated = retriever.generate(decision, metadata=sample_metadata)
        candidates.append(retriever.execute(generated))
    selected = retriever.select(sample.question, candidates, metadata=sample_metadata)
    return CandidateResults(candidates=candidates, selected=selected)


def empty_candidate_results(sample) -> CandidateResults:
    decision = RouteDecision(sample.question, sample.route, "")
    generated = GeneratedQuery(route_decision=decision, query="")
    return CandidateResults(candidates=[ExecutionResult(generated_query=generated, answer=[])])


def run_sample(i: int, sample, retriever, args) -> tuple[int, CandidateResults]:
    try:
        return i, process_sample(sample, retriever, args)
    except Exception as e:
        print(f"  sample {i} failed, using empty result: {e}")
        return i, empty_candidate_results(sample)


def main():
    args = parse_args()

    run_dir = Path(args.runs_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run dir: {run_dir}")

    # Build sample list
    if args.demo:
        samples = DEMO_QUERIES
        print(f"Using {len(samples)} demo queries")
    else:
        datasets = args.datasets if args.datasets else list(DATASET_PATHS.keys())
        samples = load_samples(
            "data/processed", datasets,
            split="test", max_samples=args.max_samples, seed=args.seed,
        )
        if args.balance:
            samples = balance_by_route(samples, args.max_per_route, args.seed)
        random.Random(args.seed).shuffle(samples)
        print(f"Using {len(samples)} samples")

    # Run pipeline
    llm = LLMClient(
        provider=args.provider,
        model=args.model,
        extra_kwargs=(
            {"chat_template_kwargs": {"enable_thinking": args.enable_thinking}}
            if args.provider == "vllm" else {}
        ),
    )
    retriever = RetrievalPipeline(llm=llm, data_dir="data/processed")

    results: list[CandidateResults | None] = [None] * len(samples)
    workers = args.concurrency if args.provider in {"openai", "anthropic", "google"} else 1
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run_sample, i, s, retriever, args) for i, s in enumerate(samples)]
        for fut in (pbar := tqdm(as_completed(futures), total=len(samples))):
            i, res = fut.result()
            results[i] = res
            pbar.set_description(f"{samples[i].source}/{samples[i].kb_id}")

    judge_llm = LLMClient(provider="openai") if args.judge else None
    metrics = evaluate(samples, cast(list[CandidateResults], results), judge_llm=judge_llm, pipeline=retriever)

    save_run(run_dir, vars(args), samples, results, metrics)


if __name__ == "__main__":
    main()
    os._exit(0)  # force-exit past any stuck Neo4j worker threads
