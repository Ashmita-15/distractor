#!/usr/bin/env python3
"""
GPU training entry point for Colab / Kaggle.

Loads prepared triplets, initializes the model, fine-tunes, saves the model,
exits. No preprocessing, no retrieval evaluation, no baseline retrieval.

Expected inputs (upload triplets_package.zip from stage 03, or place files at):
    outputs/triplets/train_triplets.jsonl
    outputs/triplets/val_triplets.jsonl

Output:
    outputs/models/finetuned_pedagogical/   (model + training_log.json)

Usage:
    python scripts/train_gpu.py [--epochs N] [--batch-size N] [--output DIR]
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.config import FINETUNE_EPOCHS, FINETUNE_BATCH_SIZE, EVAL_STEPS, MODELS_DIR
from src.utils import set_seed, print_section_header
from src.fine_tune import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU fine-tuning entry point")
    parser.add_argument("--epochs", type=int, default=FINETUNE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=FINETUNE_BATCH_SIZE)
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "finetuned_pedagogical")
    parser.add_argument("--eval-steps", type=int, default=EVAL_STEPS,
                        help="Triplet-evaluator cadence (default: %(default)s)")
    parser.add_argument("--save-steps", type=int, default=None,
                        help="Checkpoint cadence (default: same as --eval-steps)")
    parser.add_argument("--keep-all-checkpoints", action="store_true",
                        help="Keep every checkpoint (trajectory studies, e.g. Experiment 2)")
    args = parser.parse_args()

    print_section_header("GPU Training")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠ No CUDA GPU detected — training will run on CPU and be slow.")
        print("    On Colab: Runtime > Change runtime type > GPU.")

    set_seed()
    model_path = run_training(
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=None if args.keep_all_checkpoints else 2,
    )
    print(f"\n  ✅ Training complete. Model: {model_path}")


if __name__ == "__main__":
    main()
