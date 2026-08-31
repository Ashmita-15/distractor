#!/usr/bin/env python3
"""
Stage 16 — Score every valid Stage 2 candidate with the validated LLM judge.

This is the GPU half of Experiment 3.2 (R3). Scoring is deliberately separated
from selection: candidates are scored ONCE and cached to disk, after which
selection (scripts/14) and evaluation (scripts/15) are free, reproducible, and
auditable without a GPU.

Judge
    The pointwise judge validated in Stage 2 (AUC 0.842 gold-vs-random, 0.847
    gold-vs-correct-answer). Pointwise, not pairwise: pairwise judging failed
    its positive control with ~92% "A" responses regardless of content.

Independence
    This judge RANKS candidates. It must NOT also evaluate them — evaluation
    uses the gated-quality composite (embedding-based, separately validated at
    AUC 0.898/0.995). Using one instrument for both would be circular.

Inputs  : outputs/generation/stage2_generations.jsonl  (no regeneration)
Outputs : outputs/reranking/judge_candidate_scores.jsonl  (one row per candidate)

Usage
    python scripts/16_score_candidates.py --limit 5      # smoke test
    python scripts/16_score_candidates.py                # full (~880 candidates)
"""

import sys
import json
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import OUTPUT_DIR, RANDOM_SEED
from src.utils import set_seed, print_section_header
from src.reranking import hard_filter, judge_score_key
from src.llm_judge import LlamaJudge, score_pointwise

GEN_INPUT = OUTPUT_DIR / "generation" / "stage2_generations.jsonl"
OUT_DIR = OUTPUT_DIR / "reranking"
OUT_PATH = OUT_DIR / "judge_candidate_scores.jsonl"


def load_done(path: Path) -> set:
    """Keys already scored, for resume after a disconnect."""
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a truncated final line
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description="Score Stage 2 candidates with the LLM judge")
    ap.add_argument("--input", type=Path, default=GEN_INPUT)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--judge-model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--limit", type=int, default=None, help="Smoke test: first N questions")
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.input.exists():
        sys.exit(f"Stage 2 generations not found: {args.input}")

    print_section_header("STAGE 3 / EXPERIMENT 3.2 — judge scoring (R3)")
    records = [json.loads(l) for l in open(args.input) if l.strip()]
    if args.limit:
        records = records[:args.limit]
        args.out = args.out.with_name(args.out.stem + "_smoke.jsonl")
    print(f"  questions: {len(records)}")
    print(f"  judge    : {args.judge_model}")
    print(f"  output   : {args.out}")

    # ---- Build the work list from hard-filtered candidates only ----
    work = []
    for r in records:
        valid, _ = hard_filter(r)
        for idx, c in enumerate(valid):
            work.append({
                "question_id": r["question_id"],
                "valid_index": idx,
                "candidate": c["distractor"],
                "stated_misconception": c.get("stated_misconception", ""),
                "record": r,
            })
    print(f"  valid candidates to score: {len(work)}")

    done = load_done(args.out)
    todo = [w for w in work
            if judge_score_key(w["question_id"], w["candidate"]) not in done]
    if done:
        print(f"  resuming: {len(done)} already scored, {len(todo)} remaining")
    if not todo:
        print("\n  Nothing to do — all candidates already scored.")
        return

    print_section_header("Loading judge")
    judge = LlamaJudge(model_name=args.judge_model,
                       load_in_4bit=not args.no_4bit, seed=args.seed)

    print_section_header(f"Scoring {len(todo)} candidates")
    t0 = time.time()
    n_ok = n_fail = 0
    with open(args.out, "a") as fout:
        for i, w in enumerate(todo):
            res = score_pointwise(judge, w["record"], w["candidate"])
            n_ok += res["score"] is not None
            n_fail += res["score"] is None
            fout.write(json.dumps({
                "key": judge_score_key(w["question_id"], w["candidate"]),
                "question_id": w["question_id"],
                "valid_index": w["valid_index"],
                "candidate": w["candidate"],
                "stated_misconception": w["stated_misconception"],
                "score": res["score"],
                "parse_status": res["parse_status"],
                "judge_model": args.judge_model,
                "seed": args.seed,
            }) + "\n")
            fout.flush()
            if (i + 1) % 25 == 0 or i == len(todo) - 1:
                el = time.time() - t0
                print(f"    {i+1}/{len(todo)}  ok={n_ok} unscored={n_fail}  "
                      f"{el/(i+1):.1f}s/cand  elapsed {el/60:.1f} min  "
                      f"ETA {(len(todo)-i-1)*el/(i+1)/60:.1f} min")

    print_section_header("Complete")
    df = pd.DataFrame([json.loads(l) for l in open(args.out) if l.strip()])
    print(f"  scored: {len(df)} candidates over {df.question_id.nunique()} questions")
    print(f"  unscored (parse failure): {int(df.score.isna().sum())} "
          f"({df.score.isna().mean():.1%})")
    print(f"  score distribution: {df.score.value_counts().sort_index().to_dict()}")
    var = df.dropna(subset=["score"]).groupby("question_id")["score"].nunique()
    print(f"  questions where the judge distinguishes candidates "
          f"(>1 distinct score): {int((var > 1).sum())}/{len(var)} = {(var > 1).mean():.1%}")
    print("\n  (a question where every candidate scores the same cannot be "
          "re-ranked by the judge — R3 falls back to generator order there)")
    print(f"\n  [Saved] {args.out}")
    print("  Next: python scripts/14_rerank.py --judge-scores "
          f"{args.out}  then  scripts/15_evaluate_rerank.py")


if __name__ == "__main__":
    main()
