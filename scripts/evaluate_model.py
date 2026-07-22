#!/usr/bin/env python3
"""
Inference/evaluation script for a saved fine-tuned model.

Loads the saved model, embeds the held-out queries, retrieves candidates
from the corpus, computes Recall@K, MRR, nDCG, and Hit Rate, and saves
metrics.json.

This is a thin wrapper around the shared evaluation logic in
src.retrieval_experiment — identical code path to the baseline evaluation,
guaranteeing a fair comparison.

Usage:
    python scripts/evaluate_model.py [--model DIR] [--query-split test|val]
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, RESULTS_DIR
from src.utils import set_seed, print_section_header
from src.retrieval_experiment import run_retrieval_evaluation, print_metrics_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved fine-tuned model")
    parser.add_argument("--model", type=str,
                        default=str(MODELS_DIR / "finetuned_pedagogical"))
    parser.add_argument("--query-split", choices=["test", "val"], default="test")
    args = parser.parse_args()

    set_seed()
    print_section_header("Model Evaluation")
    print(f"  Model: {args.model}\n")

    result = run_retrieval_evaluation(
        model_name_or_path=args.model,
        tag="finetuned",
        query_split=args.query_split,
    )
    print_metrics_summary(result["metrics"], "Fine-tuned")

    # Canonical metrics.json for downstream consumers
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result["metrics"], f, indent=2)
    print(f"\n  [Saved] {metrics_path}")
    print("  ✅ Evaluation complete.")


if __name__ == "__main__":
    main()
