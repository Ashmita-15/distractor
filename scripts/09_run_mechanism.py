#!/usr/bin/env python3
"""
Stage 09 — Experiment 2.1 (staged): does misconception-matched context help?

Research question
    Retrieval quality can only matter for generation if misconception-matched
    exemplars actually improve generated distractors. This runs the maximal
    contrast that tests that precondition directly.

Arms (the ONLY thing that varies is exemplar selection)
    G1  Random exemplars from the whole corpus
    G4  ORACLE      — all k exemplars share a gold misconception
    G5  ANTI-ORACLE — same subject, none share a gold misconception

    G4 vs G5 is the primary contrast: both are topically constrained, so the
    comparison isolates misconception match rather than topical similarity.
    G1 provides an unconstrained reference.

    G4/G5 consult gold labels and are diagnostics, not deployable systems.

Why this before G0/G1/G2/G3
    Measured on the frozen retrievers, MNRL and pretrained MiniLM return 64%
    identical top-5 exemplars and differ in matched-exemplar status for only
    10/187 questions. The expected G3-G2 generation gap is ~0.7pp against a
    ~5.3pp detection threshold at n=187 — roughly 7x underpowered. This
    experiment instead maximises the contrast by construction, so a null
    result is informative rather than merely underpowered.

No retriever is required: all three arms select exemplars from metadata.

Usage
    python scripts/09_run_mechanism.py --limit 3 --dry-run     # inspect prompts
    python scripts/09_run_mechanism.py --limit 5               # smoke test
    python scripts/09_run_mechanism.py                         # full run
"""

import sys
import json
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.config import OUTPUT_DIR, RANDOM_SEED
from src.utils import set_seed, print_section_header
from src.retrieval_experiment import load_corpus_and_queries
from src.generation import (
    build_question_targets, select_context, annotate_exemplar_matches,
    build_prompt, parse_response, QwenGenerator, load_completed_question_ids,
    ARM_DESCRIPTIONS,
)

EXP_DIR = OUTPUT_DIR / "generation" / "exp21"
DEFAULT_ARMS = ["G1", "G4", "G5"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment 2.1 (staged mechanism test)")
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, choices=["G0", "G1", "G4", "G5"])
    ap.add_argument("--llm", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--gen-seed", type=int, default=RANDOM_SEED,
                    help="LLM sampling seed")
    ap.add_argument("--context-seed", type=int, default=RANDOM_SEED,
                    help="Exemplar-selection seed; held fixed across generation "
                         "seeds so LLM variance is isolated from context variance")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    set_seed(args.gen_seed)
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    print_section_header("EXPERIMENT 2.1 (staged): mechanism test")
    for a in args.arms:
        print(f"  {a}: {ARM_DESCRIPTIONS[a]}")
    print(f"\n  k={args.k}  M={args.m}  temp={args.temperature}  "
          f"gen_seed={args.gen_seed}  context_seed={args.context_seed}")
    print(f"  split={args.split}  generator={args.llm}")

    corpus_df, split_qdp, _ = load_corpus_and_queries(args.split)
    targets = build_question_targets(split_qdp)
    if args.limit:
        targets = targets[:args.limit]
    print(f"\n  Corpus: {len(corpus_df):,} QDPs | targets: {len(targets)}")

    # ---- Pre-compute context for every arm; report constructibility ----
    print_section_header("Context construction")
    contexts = {}
    for arm in args.arms:
        rng = np.random.RandomState(args.context_seed)
        per_q, n_ok = {}, 0
        for t in targets:
            ex, ok = select_context(arm, t, corpus_df, args.k, rng)
            if ok:
                ex = annotate_exemplar_matches(
                    ex, [g["misconception_id"] for g in t["gold_distractors"]])
                per_q[t["question_id"]] = ex
                n_ok += 1
        contexts[arm] = per_q
        matched = [sum(e["is_misconception_match"] for e in ex) for ex in per_q.values()]
        print(f"  {arm}: constructible {n_ok}/{len(targets)} "
              f"({n_ok/len(targets):.1%}) | mean matched exemplars "
              f"{np.mean(matched) if matched else 0:.2f}/{args.k}")

    common = set.intersection(*[set(c) for c in contexts.values()]) if contexts else set()
    print(f"\n  Common subset (all arms constructible): {len(common)}/{len(targets)}")
    print("  -> the primary paired contrast is computed on this subset")

    if args.dry_run:
        t0 = targets[0]
        for arm in args.arms:
            if t0["question_id"] not in contexts[arm]:
                continue
            _, user = build_prompt(t0, contexts[arm][t0["question_id"]], m=args.m)
            print(f"\n{'='*70}\nSAMPLE PROMPT — {arm} ({ARM_DESCRIPTIONS[arm]})\n{'='*70}")
            print(user[:1400])
        print("\n  Dry run complete — no generation performed.")
        return

    # ---- Load the generator ONCE and reuse across arms ----
    print_section_header("Loading generator")
    generator = QwenGenerator(
        model_name=args.llm, load_in_4bit=not args.no_4bit,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        seed=args.gen_seed,
    )

    for arm in args.arms:
        suffix = "_smoke" if args.limit else ""
        out_path = EXP_DIR / f"{arm}_seed{args.gen_seed}{suffix}.jsonl"
        done = load_completed_question_ids(out_path)
        todo = [t for t in targets
                if t["question_id"] in contexts[arm] and t["question_id"] not in done]

        print_section_header(f"ARM {arm} — {len(todo)} to generate "
                             f"({len(done)} already done)")
        if not todo:
            continue

        t0 = time.time()
        n_ok = n_failed = 0
        with open(out_path, "a") as fout:
            for i, target in enumerate(todo):
                exemplars = contexts[arm][target["question_id"]]
                system_prompt, user_prompt = build_prompt(
                    target, exemplars, m=args.m, show_misconception_labels=True)
                raw = generator.generate(system_prompt, user_prompt)
                candidates, status = parse_response(raw, m=args.m)
                n_ok += status != "failed"
                n_failed += status == "failed"

                fout.write(json.dumps({
                    "arm": arm,
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
                        "arm": arm, "llm": args.llm, "k": args.k, "m": args.m,
                        "temperature": args.temperature, "gen_seed": args.gen_seed,
                        "context_seed": args.context_seed, "split": args.split,
                    },
                }) + "\n")
                fout.flush()

                if (i + 1) % 20 == 0 or i == len(todo) - 1:
                    el = time.time() - t0
                    print(f"    {i+1}/{len(todo)}  ok={n_ok} failed={n_failed}  "
                          f"{el/(i+1):.1f}s/q  elapsed {el/60:.1f} min")
        print(f"  [Saved] {out_path}")

    print_section_header("Complete")
    print(f"  Arms generated: {', '.join(args.arms)}")
    print(f"  Next: python scripts/10_evaluate_mechanism.py")


if __name__ == "__main__":
    main()
