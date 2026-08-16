#!/usr/bin/env python3
"""
Stage 08 — Retrieval-Augmented Distractor Generation (Stage 2, minimal).

Pipeline
    test QDPs -> question-level targets (187)
              -> frozen Stage 1 retrieval (Variant B deployment query)
              -> top-k exemplars materialised from corpus
              -> retrieval-augmented prompt
              -> LLM generates M candidate distractors
              -> JSONL, written incrementally

Stage 1 is used strictly read-only: the retriever, its training code, and its
evaluation code are unchanged. The corpus keeps the full text representation
(distractors included); the query uses the deployment representation
(Subject | Construct | Question | Correct Answer), exactly as validated by the
query-representation ablation.

Usage
    # smoke test first (5 questions)
    python scripts/08_generate_distractors.py --retriever outputs/models/finetuned_mnrl --limit 5

    # full run (resumes automatically if interrupted)
    python scripts/08_generate_distractors.py --retriever outputs/models/finetuned_mnrl
"""

import sys
import json
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import QDP_TEXT_TEMPLATES, OUTPUT_DIR, MODELS_DIR, RANDOM_SEED
from src.utils import set_seed, print_section_header
from src.retrieval_experiment import load_corpus_and_queries
from src.baseline_retrieval import encode_texts, build_faiss_index, retrieve_top_k
from src.generation import (
    build_question_targets, build_query_frame, materialise_exemplars,
    annotate_exemplar_matches, build_prompt, parse_response,
    QwenGenerator, load_completed_question_ids,
)

GEN_DIR = OUTPUT_DIR / "generation"


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2: RAG distractor generation")
    ap.add_argument("--retriever", type=str,
                    default=str(MODELS_DIR / "finetuned_mnrl"),
                    help="Frozen Stage 1 MNRL retriever (path or HF name)")
    ap.add_argument("--llm", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--k", type=int, default=5, help="Retrieved exemplars per prompt")
    ap.add_argument("--m", type=int, default=5, help="Candidate distractors per question")
    ap.add_argument("--top-n", type=int, default=20,
                    help="Candidates retrieved before truncation to k")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--limit", type=int, default=None,
                    help="Smoke test: only process the first N questions")
    ap.add_argument("--no-labels", action="store_true",
                    help="Hide exemplar misconception labels (ablation)")
    ap.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantisation")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Build prompts but do not load the LLM or generate")
    args = ap.parse_args()

    set_seed(args.seed)
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (GEN_DIR / (
        "stage2_generations_smoke.jsonl" if args.limit else "stage2_generations.jsonl"))

    print_section_header("STAGE 2: Retrieval-Augmented Distractor Generation")
    print(f"  Retriever (frozen): {args.retriever}")
    print(f"  Generator:          {args.llm}")
    print(f"  k={args.k}  M={args.m}  temperature={args.temperature}  seed={args.seed}")
    print(f"  Misconception labels shown: {not args.no_labels}")
    print(f"  Output: {out_path}")

    # ---- Targets: collapse QDP-level test split to question level ----
    corpus_df, test_qdp, _ = load_corpus_and_queries("test")
    targets = build_question_targets(test_qdp)
    print(f"\n  Corpus: {len(corpus_df):,} QDPs | Test QDPs: {len(test_qdp):,} "
          f"-> {len(targets)} question-level targets")

    if args.limit:
        targets = targets[:args.limit]
        print(f"  SMOKE TEST: limited to {len(targets)} targets")

    # ---- Resume support ----
    done = load_completed_question_ids(out_path)
    if done:
        print(f"  Resuming: {len(done)} targets already complete, skipping them.")
    pending = [t for t in targets if t["question_id"] not in done]
    if not pending:
        print("\n  Nothing to do — all targets already generated.")
        return
    print(f"  Pending: {len(pending)} targets")

    # ---- Frozen Stage 1 retrieval ----
    print_section_header("Retrieval (frozen Stage 1 retriever)")
    query_df = build_query_frame(pending, QDP_TEXT_TEMPLATES["no_distractor"])
    print(f"  Example query: {query_df['text'].iloc[0][:150]!r}")

    corpus_emb = encode_texts(corpus_df["text"].tolist(),
                              model_name_or_path=str(args.retriever))
    query_emb = encode_texts(query_df["text"].tolist(),
                             model_name_or_path=str(args.retriever))
    index = build_faiss_index(corpus_emb)
    _, indices = retrieve_top_k(
        query_embeddings=query_emb, index=index, k=args.top_n,
        query_question_ids=query_df["QuestionId"].values,
        corpus_question_ids=corpus_df["QuestionId"].values,
    )
    print(f"  Retrieved: {indices.shape}")

    # ---- Generator ----
    generator = None
    if not args.dry_run:
        print_section_header("Loading generator")
        generator = QwenGenerator(
            model_name=args.llm, load_in_4bit=not args.no_4bit,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            seed=args.seed,
        )

    # ---- Generate, appending incrementally so a disconnect loses nothing ----
    print_section_header(f"Generating for {len(pending)} targets")
    t0 = time.time()
    n_ok = n_failed = 0
    with open(out_path, "a") as fout:
        for i, target in enumerate(pending):
            gold_ids = [g["misconception_id"] for g in target["gold_distractors"]]
            exemplars = materialise_exemplars(
                indices[i], corpus_df, k=args.k,
                exclude_question_id=target["question_id"],
            )
            exemplars = annotate_exemplar_matches(exemplars, gold_ids)
            system_prompt, user_prompt = build_prompt(
                target, exemplars, m=args.m,
                show_misconception_labels=not args.no_labels,
            )

            if args.dry_run:
                if i == 0:
                    print("\n----- SAMPLE PROMPT (dry run) -----")
                    print(system_prompt)
                    print("-----")
                    print(user_prompt[:2500])
                    print("----- END SAMPLE -----\n")
                continue

            raw = generator.generate(system_prompt, user_prompt)
            candidates, status = parse_response(raw, m=args.m)
            n_ok += status != "failed"
            n_failed += status == "failed"

            record = {
                "question_id": target["question_id"],
                "subject": target["subject"],
                "construct": target["construct"],
                "question_text": target["question_text"],
                "correct_answer": target["correct_answer"],
                "gold_distractors": target["gold_distractors"],
                "retrieved": exemplars,
                "n_exemplars_matched": sum(e["is_misconception_match"] for e in exemplars),
                "prompt": {"system": system_prompt, "user": user_prompt},
                "raw_output": raw,
                "parse_status": status,
                "candidates": candidates,
                "config": {
                    "retriever": str(args.retriever), "llm": args.llm,
                    "k": args.k, "m": args.m, "temperature": args.temperature,
                    "seed": args.seed, "show_labels": not args.no_labels,
                    "query_template": "no_distractor",
                },
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

            if (i + 1) % 10 == 0 or i == len(pending) - 1:
                el = time.time() - t0
                print(f"    {i+1}/{len(pending)}  ok={n_ok} failed={n_failed}  "
                      f"{el/(i+1):.1f}s/question  elapsed {el/60:.1f} min")

    if args.dry_run:
        print("  Dry run complete — prompts built, no generation performed.")
        return

    print_section_header("Complete")
    print(f"  Generated: {n_ok} | parse failures: {n_failed}")
    print(f"  [Saved] {out_path}")
    if n_failed:
        print("  Note: raw_output is retained for every record, so failed parses "
              "can be recovered without regenerating.")


if __name__ == "__main__":
    main()
