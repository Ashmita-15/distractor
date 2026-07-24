#!/usr/bin/env python3
"""
Stage 04 — Fine-tune the SentenceTransformer with triplet loss.

Single responsibility: load prepared triplets, fine-tune, save the best
checkpoint and training log. Does NOT run EDA, baseline retrieval, metric
computation, or triplet construction.

Requires only:
    outputs/triplets/train_triplets.jsonl
    outputs/triplets/val_triplets.jsonl
(produced by scripts/03_create_triplets.py — or uploaded to the cloud host)

Usage:
    python scripts/04_train.py [--epochs N] [--batch-size N]
                               [--output DIR] [--smoke-test]
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FINETUNE_EPOCHS, FINETUNE_BATCH_SIZE, EVAL_STEPS, MODELS_DIR
from src.utils import set_seed, print_section_header
from src.fine_tune import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune SentenceTransformer on triplets")
    parser.add_argument("--epochs", type=int, default=FINETUNE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=FINETUNE_BATCH_SIZE)
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "finetuned_pedagogical")
    parser.add_argument("--eval-steps", type=int, default=EVAL_STEPS,
                        help="Triplet-evaluator cadence (default: %(default)s)")
    parser.add_argument("--save-steps", type=int, default=None,
                        help="Checkpoint cadence (default: same as --eval-steps)")
    parser.add_argument("--keep-all-checkpoints", action="store_true",
                        help="Keep every checkpoint (trajectory studies, e.g. Experiment 2)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Debug only: train on 64 triplets for 1 epoch to verify the stage runs")
    args = parser.parse_args()

    set_seed()
    print_section_header("STAGE 04: Fine-tuning")

    model_path = run_training(
        output_dir=args.output,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=args.batch_size,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=None if args.keep_all_checkpoints else 2,
        max_train_triplets=64 if args.smoke_test else None,
    )
    print(f"\n  ✅ Stage 04 complete. Model: {model_path}")


if __name__ == "__main__":
    main()
