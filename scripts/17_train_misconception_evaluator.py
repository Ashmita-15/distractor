#!/usr/bin/env python3
"""
Stage 17 — Train the independent misconception-alignment evaluator.

Splits (reuses the existing question-disjoint splits; no new partitioning):
    train QDPs (3,507)  -> fit the ranker
    val   QDPs (432)    -> early stopping / checkpoint selection
    test  QDPs (431)    -> NEVER seen; used in script 18 as the reference anchor

Contamination checks run before training and abort on failure:
    - no misconception description appears in any evaluator input
    - evaluator training questions are disjoint from the application questions
    - the base encoder differs from the Stage 1 retriever's

Usage
    python scripts/17_train_misconception_evaluator.py --epochs 3
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import MISCONCEPTION_CSV, RESULTS_DIR, OUTPUT_DIR, RANDOM_SEED
from src.utils import set_seed, print_section_header
from src.misconception_evaluator import (
    build_training_pairs, build_queries_from_qdp, train_evaluator,
    assert_no_label_leakage, assert_no_question_overlap, check_independence,
)

OUT_DIR = OUTPUT_DIR / "misconception_evaluator"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    set_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print_section_header("MISCONCEPTION EVALUATOR — TRAINING")
    mm = pd.read_csv(MISCONCEPTION_CSV)
    tr = pd.read_csv(RESULTS_DIR / "train_qdp.csv")
    va = pd.read_csv(RESULTS_DIR / "val_qdp.csv")
    te = pd.read_csv(RESULTS_DIR / "test_qdp.csv")
    print(f"  label space : {len(mm)} misconceptions")
    print(f"  train       : {len(tr)} QDPs / {tr.QuestionId.nunique()} questions")
    print(f"  dev (val)   : {len(va)} QDPs / {va.QuestionId.nunique()} questions")
    print(f"  held out    : {len(te)} QDPs / {te.QuestionId.nunique()} questions (test)")

    # ---- Mandatory contamination checks ----
    print_section_header("Contamination checks")
    checks = {}
    checks["independence"] = check_independence(args.base_model)
    print(f"  base encoder distinct from Stage 1 : "
          f"{checks['independence']['distinct_base_encoder']} "
          f"({args.base_model})")
    checks["question_overlap_train_vs_test"] = assert_no_question_overlap(tr, te)
    checks["question_overlap_dev_vs_test"] = assert_no_question_overlap(va, te)
    print(f"  train/test question overlap        : "
          f"{checks['question_overlap_train_vs_test']['overlap']}")
    checks["label_leakage_train"] = assert_no_label_leakage(build_queries_from_qdp(tr), mm)
    checks["label_leakage_test"] = assert_no_label_leakage(build_queries_from_qdp(te), mm)
    print(f"  queries containing a label name    : 0 "
          f"({checks['label_leakage_train']['queries_checked']} train, "
          f"{checks['label_leakage_test']['queries_checked']} test checked)")
    print("  generator self-reported misconception is not an input (by construction)")

    train_pairs = build_training_pairs(tr, mm)
    dev_pairs = build_training_pairs(va, mm)
    print(f"\n  training pairs: {len(train_pairs)} | dev pairs: {len(dev_pairs)}")

    print_section_header("Training")
    print(f"  base={args.base_model}  epochs={args.epochs}  "
          f"batch={args.batch_size}  seed={args.seed}")
    model_path = train_evaluator(
        train_pairs, dev_pairs, output_dir=OUT_DIR,
        base_model=args.base_model, epochs=args.epochs,
        batch_size=args.batch_size, seed=args.seed,
    )

    meta = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "n_misconception_classes": int(len(mm)),
        "n_train_pairs": len(train_pairs),
        "n_dev_pairs": len(dev_pairs),
        "n_heldout_test_qdps": int(len(te)),
        "contamination_checks": checks,
        "model_path": str(model_path),
    }
    (OUT_DIR / "validation_metrics.json").write_text(json.dumps(meta, indent=2))
    print_section_header("Complete")
    print(f"  model: {model_path}")
    print(f"  [Saved] {OUT_DIR / 'validation_metrics.json'}")
    print("  Next: python scripts/18_evaluate_misconception_alignment.py")


if __name__ == "__main__":
    main()
