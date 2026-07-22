#!/usr/bin/env python3
"""
Stage 05 — Evaluate a retrieval model and (optionally) compare to baseline.

Single responsibility: retrieval evaluation of one embedding model
(typically the fine-tuned checkpoint downloaded from Colab), with an
optional full comparison against the saved baseline artifacts.

Usage:
    python scripts/05_evaluate.py                       # fine-tuned model, default path
    python scripts/05_evaluate.py --model PATH_OR_NAME  # any model
    python scripts/05_evaluate.py --compare             # + comparison vs baseline
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR
from src.utils import set_seed, print_section_header
from src.retrieval_experiment import run_retrieval_evaluation, print_metrics_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a retrieval model")
    parser.add_argument("--model", type=str,
                        default=str(MODELS_DIR / "finetuned_pedagogical"),
                        help="Model path or HuggingFace name")
    parser.add_argument("--tag", type=str, default="finetuned",
                        help="Artifact prefix for saved outputs")
    parser.add_argument("--query-split", choices=["test", "val"], default="test")
    parser.add_argument("--compare", action="store_true",
                        help="Run full comparison against saved baseline artifacts")
    args = parser.parse_args()

    set_seed()
    print_section_header("STAGE 05: Retrieval Evaluation")
    print(f"  Model: {args.model}\n")

    result = run_retrieval_evaluation(
        model_name_or_path=args.model,
        tag=args.tag,
        query_split=args.query_split,
    )
    print_metrics_summary(result["metrics"], args.tag)

    if args.compare:
        print_section_header("Comparison: Baseline vs Fine-tuned")
        from src.comparison import run_comparison
        run_comparison(baseline_tag="baseline", finetuned_tag=args.tag)

    print("\n  ✅ Stage 05 complete.")


if __name__ == "__main__":
    main()
