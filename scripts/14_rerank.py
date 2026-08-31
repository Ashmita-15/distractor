#!/usr/bin/env python3
"""
Stage 14 — Apply Stage 3 re-ranking to existing Stage 2 generations.

Reads the Stage 2 output (no regeneration), applies the deterministic hard
filter, then emits one selection per question under each strategy:

    R0_first      first valid candidate (Stage 2 default; the baseline)
    R1_random     random valid candidate (controls for "any selection")
    R2_heuristic  heuristic plausibility ranking (no gold, no tuning)
    R4_oracle     best by gold similarity (ceiling; diagnostic only)

Every strategy selects from the SAME filtered candidate list, so selection
rule is the only variable.

Usage
    python scripts/14_rerank.py
    python scripts/14_rerank.py --input outputs/generation/stage2_generations.jsonl --k 1
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR, RESULTS_DIR, RANDOM_SEED
from src.utils import set_seed, print_section_header
from src.reranking import (
    hard_filter, select, heuristic_score, judge_score_key,
    STRATEGIES, STRATEGY_R3, STRATEGY_DESCRIPTIONS,
)

DEFAULT_INPUT = OUTPUT_DIR / "generation" / "stage2_generations.jsonl"
OUT_DIR = OUTPUT_DIR / "reranking"


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3: apply re-ranking strategies")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--k", type=int, default=1,
                    help="Distractors to select per question (1 = primary analysis)")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--judge-scores", type=Path, default=None,
                    help="Cached judge scores from scripts/16_score_candidates.py; "
                         "enables the R3_llm strategy (Experiment 3.2)")
    args = ap.parse_args()

    set_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.input.exists():
        sys.exit(f"Stage 2 generations not found: {args.input}")

    # R3 is included only when cached judge scores are supplied, so Experiment
    # 3.1 remains reproducible without a GPU.
    judge_scores = None
    strategies = list(STRATEGIES)
    if args.judge_scores:
        if not args.judge_scores.exists():
            sys.exit(f"Judge scores not found: {args.judge_scores}")
        judge_scores = {}
        for line in open(args.judge_scores):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("score") is not None:
                judge_scores[row["key"]] = float(row["score"])
        strategies.insert(3, STRATEGY_R3)

    print_section_header("STAGE 3 — RE-RANKING")
    records = [json.loads(l) for l in open(args.input) if l.strip()]
    print(f"  input: {args.input}")
    print(f"  questions: {len(records)}  |  k = {args.k}")
    if judge_scores is not None:
        print(f"  judge scores loaded: {len(judge_scores)} candidates "
              f"(Experiment 3.2 enabled)")
    for s in strategies:
        print(f"    {s:14s} {STRATEGY_DESCRIPTIONS[s]}")

    # ---- Hard filter (shared by every strategy) ----
    print_section_header("Hard filter (deterministic)")
    filtered, stats_rows = {}, []
    for r in records:
        valid, stats = hard_filter(r)
        filtered[r["question_id"]] = valid
        stats_rows.append({"question_id": r["question_id"], **stats})
    fs = pd.DataFrame(stats_rows)
    fs.to_csv(OUT_DIR / "filter_stats.csv", index=False)

    tot = int(fs.n_input.sum())
    print(f"  candidates in            : {tot}")
    for col, label in [("n_empty", "empty"), ("n_correct_answer", "= correct answer"),
                       ("n_duplicate", "duplicate"), ("n_valid", "VALID")]:
        n = int(fs[col].sum())
        print(f"  {label:24s} : {n:4d} ({n/tot:.1%})")
    print(f"  valid per question       : mean {fs.n_valid.mean():.2f}, "
          f"min {int(fs.n_valid.min())}, "
          f"questions with 0 valid: {int((fs.n_valid == 0).sum())}")

    # ---- Apply each strategy ----
    print_section_header("Selection")
    out_rows = []
    for strategy in strategies:
        rng = np.random.RandomState(args.seed)   # fresh per strategy = reproducible
        n_sel = 0
        for r in records:
            valid = filtered[r["question_id"]]
            chosen = select(strategy, r, valid, rng=rng, k=args.k,
                            judge_scores=judge_scores)
            if not chosen:
                continue
            n_sel += 1
            out_rows.append({
                "strategy": strategy,
                "question_id": r["question_id"],
                "correct_answer": r["correct_answer"],
                "selected": json.dumps([c["distractor"] for c in chosen]),
                "selected_misconceptions": json.dumps(
                    [c.get("stated_misconception", "") for c in chosen]),
                "n_valid_available": len(valid),
                "heuristic_score": round(heuristic_score(
                    chosen[0]["distractor"], r["correct_answer"])["heuristic_score"], 4),
            })
        print(f"  {strategy:14s} selections: {n_sel}/{len(records)}")

    sel = pd.DataFrame(out_rows)
    sel_path = OUT_DIR / f"selections_k{args.k}.csv"
    sel.to_csv(sel_path, index=False)
    print(f"\n  [Saved] {sel_path}")

    # ---- Sanity: do strategies actually differ? ----
    print_section_header("Manipulation check")
    piv = sel.pivot_table(index="question_id", columns="strategy",
                          values="selected", aggfunc="first")
    for s in strategies:
        if s == "R0_first" or s not in piv.columns:
            continue
        common = piv[["R0_first", s]].dropna()
        diff = (common["R0_first"] != common[s]).mean()
        print(f"  {s:14s} differs from R0 on {diff:.1%} of questions")
    print("\n  (a strategy that never differs from R0 cannot show an effect)")

    print("\n  ✅ Re-ranking complete. Next: scripts/15_evaluate_rerank.py")


if __name__ == "__main__":
    main()
