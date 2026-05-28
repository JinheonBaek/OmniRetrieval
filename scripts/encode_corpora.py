#!/usr/bin/env python3
"""Pre-encode BEIR corpus documents and save embeddings to disk cache.

Usage:
    # Encode all BEIR corpora
    python -m scripts.encode_corpora

    # Encode specific corpora only
    python -m scripts.encode_corpora --corpora nfcorpus scifact

    # Use a different model
    python -m scripts.encode_corpora --model sentence-transformers/all-mpnet-base-v2
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from sentence_transformers import SentenceTransformer

from src.data.beir_corpora import extract_corpus_sentences


def main():
    parser = argparse.ArgumentParser(description="Pre-encode BEIR corpus embeddings")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model name")
    parser.add_argument("--data-dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--cache-dir", default="data/cache", help="Cache directory")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--corpora", nargs="*", help="Specific corpora to encode (default: all)")
    args = parser.parse_args()

    corpus_dir = Path(args.data_dir) / "beir" / "corpus"
    model_name_safe = args.model.replace("/", "_")
    output_dir = Path(args.cache_dir) / "beir" / model_name_safe
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover corpus files from disk
    corpus_files = sorted(corpus_dir.glob("*_corpus.jsonl"))
    if args.corpora:
        corpus_files = [f for f in corpus_files if any(c in f.name for c in args.corpora)]

    model = SentenceTransformer(args.model)

    for corpus_path in corpus_files:
        # e.g., "nfcorpus_corpus.jsonl" -> "nfcorpus", "nq_test_corpus.jsonl" -> "nq_test"
        kb_id = corpus_path.name.removesuffix("_corpus.jsonl")

        doc_ids_path = output_dir / f"{kb_id}_doc_ids.json"
        embeddings_path = output_dir / f"{kb_id}_embeddings.pt"

        if doc_ids_path.exists() and embeddings_path.exists():
            print(f"[{kb_id}] Cache already exists, skipping.")
            continue

        corpus = {}
        with open(corpus_path) as f:
            for line in f:
                doc = json.loads(line)
                corpus[doc["_id"]] = {"title": doc.get("title", ""), "text": doc.get("text", "")}

        doc_ids = list(corpus.keys())
        texts = extract_corpus_sentences([corpus[d] for d in doc_ids], sep=" ")

        print(f"[{kb_id}] Encoding {len(doc_ids)} documents...")
        embeddings = model.encode(
            texts, batch_size=args.batch_size, show_progress_bar=True,
            normalize_embeddings=True, convert_to_numpy=True,
        )
        embeddings = torch.from_numpy(embeddings)

        with open(doc_ids_path, "w") as f:
            json.dump(doc_ids, f)
        torch.save(embeddings, embeddings_path)
        print(f"[{kb_id}] Saved to {output_dir}")


if __name__ == "__main__":
    main()
