"""
Preprocess BEIR datasets into UnifiedSample JSONL files.

Reads raw BEIR data (queries.jsonl + qrels TSVs) and converts each query
into a UnifiedSample. For MSMARCO the dev split is used instead of test
because the official test labels are not publicly available.
Corpus files are copied to the output directory to be self-contained.

For NQ, the train split comes from 'nq-train' and the test split from 'nq'.
The two NQ folders have different corpora with colliding document IDs, so
they are stored as separate corpus files. 

In each sample, metadata["corpus_file"] records which corpus to use for
document lookup. All other datasets share a single corpus across splits.

Output layout:
    data/processed/beir/
    ├── corpus/
    │   ├── <dataset>_corpus.jsonl   (one per dataset, two for NQ)
    │   └── ...
    ├── <dataset>.jsonl              (UnifiedSample records)
    └── ...
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.schema import RouteType, UnifiedSample, save_jsonl

RAW_DIR = Path("data/raw/beir")
OUT_DIR = Path("data/processed/beir")

# Mapping from dataset name → list of (split_name, qrels_folder, qrels_file).
# NQ is special: train lives in "nq-train/" and test lives in "nq/".
DATASETS: dict[str, list[tuple[str, str, str]]] = {
    "fever":    [("train", "fever",    "train.tsv"), ("test", "fever",    "test.tsv")],
    "fiqa":     [("train", "fiqa",     "train.tsv"), ("test", "fiqa",     "test.tsv")],
    "msmarco":  [("train", "msmarco",  "train.tsv"), ("test", "msmarco",  "dev.tsv" )],
    "scifact":  [("train", "scifact",  "train.tsv"), ("test", "scifact",  "test.tsv")],
    "hotpotqa": [("train", "hotpotqa", "train.tsv"), ("test", "hotpotqa", "test.tsv")],
    "nfcorpus": [("train", "nfcorpus", "train.tsv"), ("test", "nfcorpus", "test.tsv")],
    "nq":       [("train", "nq-train", "train.tsv"), ("test", "nq",       "test.tsv")],
}


def load_queries(dataset_folder: Path) -> dict[str, str]:
    """Load queries.jsonl → {query_id: question_text}."""
    queries: dict[str, str] = {}
    queries_path = dataset_folder / "queries.jsonl"
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            queries[obj["_id"]] = obj["text"]
    return queries


def load_qrels(qrels_path: Path) -> dict[str, list[str]]:
    """Load a qrels TSV → {query_id: [corpus_id, ...]} (positive only, score > 0)."""
    qrels: dict[str, list[str]] = defaultdict(list)
    with open(qrels_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if int(row["score"]) > 0:
                qrels[row["query-id"]].append(row["corpus-id"])
    return qrels


def copy_corpus(src: Path, dst: Path) -> None:
    """Copy a corpus file if the destination does not already exist."""
    if dst.exists():
        print(f"  Corpus already exists: {dst}")
        return
    shutil.copy2(src, dst)
    print(f"  Copied corpus → {dst}")


def process_split(
    dataset_name: str,
    split: str,
    folder: str,
    qrels_file: str,
    corpus_file: str,
) -> list[UnifiedSample]:
    """Convert one (dataset, split) pair into a list of UnifiedSample."""
    dataset_folder = RAW_DIR / folder
    qrels_path = dataset_folder / "qrels" / qrels_file

    assert qrels_path.exists(), f"qrels file not found: {qrels_path}"

    queries = load_queries(dataset_folder)
    qrels = load_qrels(qrels_path)

    samples: list[UnifiedSample] = []
    for query_id, corpus_ids in sorted(qrels.items()):
        question = queries.get(query_id)
        if question is None:
            continue
        samples.append(
            UnifiedSample(
                id=f"{dataset_name}-{split}-{query_id}",
                source=dataset_name,
                split=split,
                question=question,
                route=RouteType.SEARCH,
                kb_id=dataset_name,
                target_query=None,
                gold_answer=corpus_ids,
                metadata={"corpus_file": corpus_file},
            )
        )
    return samples


def corpus_filename(dataset_name: str, folder: str) -> str:
    """Determine the output corpus filename for a given dataset/folder pair."""
    return (
        f"nq_{'train' if folder == 'nq-train' else 'test'}_corpus.jsonl"
        if dataset_name == "nq"  # NQ has two distinct corpora; name them by the raw folder.
        else f"{dataset_name}_corpus.jsonl"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus_dir = OUT_DIR / "corpus"
    corpus_dir.mkdir(exist_ok=True)

    total = 0
    copied_corpora: set[str] = set()

    for dataset_name, splits in DATASETS.items():
        samples: list[UnifiedSample] = []
        for split, folder, qrels_file in splits:
            # Copy corpus (once per unique raw folder)
            c_filename = corpus_filename(dataset_name, folder)
            if c_filename not in copied_corpora:
                src_corpus = RAW_DIR / folder / "corpus.jsonl"
                copy_corpus(src_corpus, corpus_dir / c_filename)
                copied_corpora.add(c_filename)

            split_samples = process_split(
                dataset_name, split, folder, qrels_file, corpus_file=c_filename,
            )
            samples.extend(split_samples)
            print(f"  {dataset_name}/{split}: {len(split_samples)} samples")

        assert samples, f"No samples found for {dataset_name}"
        out_path = OUT_DIR / f"{dataset_name}.jsonl"
        save_jsonl(samples, str(out_path))
        total += len(samples)

    print(f"\nTotal: {total} samples written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
