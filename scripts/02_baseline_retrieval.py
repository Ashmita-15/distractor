#!/usr/bin/env python3
"""
Stage 02 — Baseline semantic retrieval and evaluation.

Single responsibility: evaluate the pretrained (not fine-tuned)
SentenceTransformer under the standard retrieval protocol and persist all
artifacts needed for the later comparison.

Protocol: corpus = train+val QDPs, queries = test QDPs, exact cosine
retrieval, relevance = shared MisconceptionId.

Usage:
    python scripts/02_baseline_retrieval.py [--query-split test|val]
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BASELINE_MODEL_NAME
from src.utils import set_seed, print_section_header
from src.retrieval_experiment import run_retrieval_evaluation, print_metrics_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline retrieval evaluation")
    parser.add_argument("--query-split", choices=["test", "val"], default="test",
                        help="Query split (default: test — the standard protocol)")
    args = parser.parse_args()

    set_seed()
    print_section_header("STAGE 02: Baseline Semantic Retrieval")
    print(f"  Model: {BASELINE_MODEL_NAME}\n")

    result = run_retrieval_evaluation(
        model_name_or_path=BASELINE_MODEL_NAME,
        tag="baseline",
        query_split=args.query_split,
    )
    print_metrics_summary(result["metrics"], "Baseline")
    print("\n  ✅ Baseline evaluation complete.")


if __name__ == "__main__":
    main()
