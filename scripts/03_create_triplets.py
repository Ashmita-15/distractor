#!/usr/bin/env python3
"""
Stage 03 — Triplet construction.

Single responsibility: build (anchor, positive, negative) training and
validation triplets from the prepared splits and save them to disk, so the
training stage (04_train.py / train_gpu.py) needs nothing but these files.

Also writes triplets_package.zip for one-file upload to Colab/Kaggle.

Usage:
    python scripts/03_create_triplets.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RANDOM_SEED, TRIPLETS_DIR
from src.utils import set_seed, print_section_header
from src.retrieval_experiment import load_split
from src.triplet_construction import (
    construct_triplets, validate_triplets, save_triplets, package_triplets,
)


def main() -> None:
    set_seed()
    print_section_header("STAGE 03: Triplet Construction")

    train_qdp = load_split("train")
    val_qdp = load_split("val")

    print("Constructing training triplets (in-subject negatives)...")
    train_triplets = construct_triplets(
        train_qdp, strategy="in_subject", negatives_per_anchor=3, seed=RANDOM_SEED,
    )

    print("\nConstructing validation triplets (in-subject negatives)...")
    val_triplets = construct_triplets(
        val_qdp, strategy="in_subject", negatives_per_anchor=2, seed=RANDOM_SEED + 1,
    )

    print("\nSample triplets for manual validation:")
    validate_triplets(train_triplets, n_samples=3)

    save_triplets(train_triplets, "train_triplets.jsonl")
    save_triplets(val_triplets, "val_triplets.jsonl")
    package_triplets(["train_triplets.jsonl", "val_triplets.jsonl"])

    print(f"\n  ✅ Triplets saved to {TRIPLETS_DIR}")
    print(f"      Training:   {len(train_triplets):,}")
    print(f"      Validation: {len(val_triplets):,}")


if __name__ == "__main__":
    main()
