#!/usr/bin/env python3
"""
Stage 01 — Dataset preparation.

Single responsibility: turn the raw Eedi CSVs into leak-free QDP splits.

    raw CSVs -> validate -> melt to QDPs -> text representation
             -> question-level stratified train/val/test split
             -> outputs/results/{train,val,test}_qdp.csv

Optionally regenerates the research EDA figures with --eda.

Usage:
    python scripts/01_prepare_dataset.py [--eda]
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR, FIGURES_DIR
from src.utils import set_seed, save_results, print_section_header
from src.data_loader import (
    load_raw_data, validate_data_integrity, melt_to_qdp,
    create_text_representation, get_misconception_stats,
)
from src.data_preparation import create_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare QDP dataset splits")
    parser.add_argument("--eda", action="store_true",
                        help="Also regenerate EDA figures and statistics")
    args = parser.parse_args()

    set_seed()
    print_section_header("STAGE 01: Dataset Preparation")

    print("Loading raw data...")
    train_df, test_df, misconception_df = load_raw_data()

    print("\nValidating data integrity...")
    integrity = validate_data_integrity(train_df, misconception_df)
    if not integrity["all_valid"]:
        sys.exit("Data integrity checks failed — aborting.")

    print("\nMelting to Question-Distractor Pairs (QDPs)...")
    qdp_df = melt_to_qdp(train_df, misconception_df, keep_unlabeled=False)

    print("\nCreating text representations...")
    qdp_df = create_text_representation(qdp_df)

    print("\nTop misconceptions:")
    stats = get_misconception_stats(qdp_df)
    print(stats.head(10)[["MisconceptionId", "count", "misconception_name"]].to_string(index=False))

    print("\nCreating question-level stratified splits...")
    train_qdp, val_qdp, test_qdp = create_splits(qdp_df)

    train_qdp.to_csv(RESULTS_DIR / "train_qdp.csv", index=False)
    val_qdp.to_csv(RESULTS_DIR / "val_qdp.csv", index=False)
    test_qdp.to_csv(RESULTS_DIR / "test_qdp.csv", index=False)
    print(f"\n  ✅ Splits saved to {RESULTS_DIR}")

    if args.eda:
        print_section_header("Research EDA")
        from src.eda import run_full_eda
        eda_stats = run_full_eda(train_df, qdp_df, misconception_df)
        save_results(eda_stats, "eda_statistics.json", RESULTS_DIR)
        print(f"  ✅ EDA figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
