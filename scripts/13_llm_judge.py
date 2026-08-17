#!/usr/bin/env python3
"""
Stage 13 — LLM-as-judge evaluation, gated by a positive control.

Human evaluation is unavailable, so this substitutes for it on the constructs
no automatic metric can measure (plausibility, educational usefulness) and on
the near-miss blind spot that every automatic metric failed (AUC ~0.56).

STEP 1 — VALIDATION (gate)
    The judge is tested on pairs whose correct answer is known by
    construction. It must clear a pre-set bar or the script stops WITHOUT
    scoring the experiment. This is the same discipline that exposed Exact
    Match as blind, applied to the judge itself.

        C1  gold distractor  vs  random corpus distractor   -> gold better
        C2  gold distractor  vs  the correct answer         -> gold better
        C3  gold distractor  vs  digit-perturbed gold       -> gold better
                                                    (the automatic blind spot)

    Every pair is judged in BOTH orders, so position bias is measured rather
    than assumed away.

    Bar:  C1 accuracy >= 0.80  AND  C2 accuracy >= 0.80
          AND order-inconsistency <= 0.30
    C3 is reported but NOT gated: no existing metric passes it, so it is
    treated as a capability probe rather than a requirement.

STEP 2 — EXPERIMENT (only if the gate passes)
    Blind pairwise comparison of arms on the shared questions, plus a GOLD
    anchor arm giving the interpretable headline "generated distractors are
    preferred over teacher-written ones X% of the time."

Usage
    python scripts/13_llm_judge.py --validate-only --limit 40   # gate only
    python scripts/13_llm_judge.py                              # gate + experiment
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
from scipy import stats

from src.config import RESULTS_DIR, OUTPUT_DIR, RANDOM_SEED
from src.utils import set_seed, print_section_header
from src.retrieval_experiment import load_corpus_and_queries
from src.generation import build_question_targets, ARM_DESCRIPTIONS
from src.gen_evaluation import normalise_answer
from src.llm_judge import LlamaJudge, judge_pair_both_orders

GEN_DIR = OUTPUT_DIR / "generation" / "exp21"
OUT_DIR = RESULTS_DIR / "llm_judge"

C1_BAR = 0.80
C2_BAR = 0.80
INCONSISTENCY_BAR = 0.30

_NUM = __import__("re").compile(r"-?\d+\.?\d*")


def perturb(s: str) -> str:
    """Digit-perturb a numeric answer (the automatic-metric blind spot)."""
    m = _NUM.search(str(s))
    if not m:
        return str(s) + " (altered)"
    try:
        v = float(m.group())
    except ValueError:
        return str(s) + " (altered)"
    new = v + (1 if abs(v) < 100 else 10)
    new_s = str(int(new)) if float(new).is_integer() else f"{new:g}"
    return str(s)[:m.start()] + new_s + str(s)[m.end():]


# =============================================================================
# STEP 1 — validation
# =============================================================================

def run_validation(judge, targets, corpus_df, limit, seed) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    corpus_distractors = corpus_df["DistractorText"].tolist()
    usable = [t for t in targets if t["gold_distractors"]][:limit]

    rows = []
    t0 = time.time()
    for i, t in enumerate(usable):
        gold = t["gold_distractors"][0]["distractor"]
        cases = [
            ("C1_gold_vs_random", corpus_distractors[rng.randint(len(corpus_distractors))]),
            ("C2_gold_vs_correct", t["correct_answer"]),
            ("C3_gold_vs_perturbed", perturb(gold)),
        ]
        for cond, bad in cases:
            if normalise_answer(bad) == normalise_answer(gold):
                continue  # degenerate pair, skip
            res = judge_pair_both_orders(judge, t, good=str(gold), bad=str(bad))
            rows.append({"condition": cond, "question_id": t["question_id"],
                         "good": str(gold), "bad": str(bad), **res})
        if (i + 1) % 10 == 0 or i == len(usable) - 1:
            el = time.time() - t0
            print(f"    {i+1}/{len(usable)} questions  {el/(i+1):.1f}s/q  "
                  f"elapsed {el/60:.1f} min")
    return pd.DataFrame(rows)


def summarise_validation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond, grp in df.groupby("condition"):
        n = len(grp)
        scored = grp[grp.outcome.isin(["correct", "incorrect", "tie"])]
        acc = (scored.outcome == "correct").mean() if len(scored) else np.nan
        # binomial test of accuracy against chance (0.5)
        n_correct = int((scored.outcome == "correct").sum())
        n_dec = int((scored.outcome.isin(["correct", "incorrect"])).sum())
        p = stats.binomtest(n_correct, n_dec, 0.5).pvalue if n_dec else 1.0
        rows.append({
            "condition": cond, "n_pairs": n,
            "accuracy": round(float(acc), 4) if acc == acc else np.nan,
            "inconsistent": round(float((grp.outcome == "inconsistent").mean()), 4),
            "tie": round(float((grp.outcome == "tie").mean()), 4),
            "unparsed": round(float((grp.outcome == "unparsed").mean()), 4),
            "p_vs_chance": f"{p:.3g}",
        })
    return pd.DataFrame(rows)


# =============================================================================
# STEP 2 — experiment
# =============================================================================

def load_arms() -> dict:
    out = {}
    for path in sorted(GEN_DIR.glob("*_seed*.jsonl")):
        if path.name.endswith("_smoke.jsonl"):
            continue
        arm = path.name.split("_")[0]
        for r in (json.loads(l) for l in open(path) if l.strip()):
            out.setdefault(arm, {})[r["question_id"]] = r
    return out


def best_candidate(record) -> str:
    """First valid candidate (not empty, not the correct answer)."""
    correct = normalise_answer(record["correct_answer"])
    for c in record.get("candidates", []):
        d = str(c.get("distractor", "")).strip()
        if d and normalise_answer(d) != correct:
            return d
    return ""


def run_experiment(judge, arms, pairs, limit) -> pd.DataFrame:
    rows = []
    for arm_a, arm_b in pairs:
        if arm_a != "GOLD" and arm_a not in arms:
            continue
        if arm_b not in arms:
            continue
        qs = set(arms[arm_b]) & (set(arms[arm_a]) if arm_a != "GOLD" else set(arms[arm_b]))
        qs = sorted(qs)[:limit]
        print(f"\n  {arm_b} vs {arm_a}: {len(qs)} questions")
        t0 = time.time()
        for i, qid in enumerate(qs):
            rb = arms[arm_b][qid]
            cand_b = best_candidate(rb)
            if arm_a == "GOLD":
                cand_a = str(rb["gold_distractors"][0]["distractor"])
            else:
                cand_a = best_candidate(arms[arm_a][qid])
            if not cand_a or not cand_b:
                continue
            # 'good' slot holds arm_b so 'correct' == arm_b preferred
            res = judge_pair_both_orders(judge, rb, good=cand_b, bad=cand_a)
            rows.append({"comparison": f"{arm_b} vs {arm_a}", "question_id": qid,
                         "arm_b_candidate": cand_b, "arm_a_candidate": cand_a, **res})
            if (i + 1) % 20 == 0 or i == len(qs) - 1:
                el = time.time() - t0
                print(f"    {i+1}/{len(qs)}  {el/(i+1):.1f}s/q  {el/60:.1f} min")
    return pd.DataFrame(rows)


def summarise_experiment(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comp, grp in df.groupby("comparison"):
        dec = grp[grp.outcome.isin(["correct", "incorrect"])]
        n_b = int((dec.outcome == "correct").sum())
        n_a = int((dec.outcome == "incorrect").sum())
        p = stats.binomtest(n_b, n_b + n_a, 0.5).pvalue if (n_b + n_a) else 1.0
        rows.append({
            "comparison": comp, "n_pairs": len(grp),
            "n_decisive": n_b + n_a,
            "treatment_win_rate": round(n_b / (n_b + n_a), 4) if (n_b + n_a) else np.nan,
            "ties": round(float((grp.outcome == "tie").mean()), 4),
            "inconsistent": round(float((grp.outcome == "inconsistent").mean()), 4),
            "p_vs_chance": f"{p:.4g}",
            "significant_0.05": "yes" if p < 0.05 else "no",
        })
    return pd.DataFrame(rows)


# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--limit", type=int, default=40,
                    help="Questions per validation condition / per comparison")
    ap.add_argument("--exp-limit", type=int, default=80,
                    help="Questions per experimental comparison")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Score the experiment even if the gate fails (NOT recommended)")
    args = ap.parse_args()

    set_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print_section_header("LLM-AS-JUDGE — positive-control gate")
    print(f"  Judge: {args.judge_model}")
    print("  Generator was Qwen2.5-7B-Instruct -> different family, "
          "so self-preference is avoided.")
    print(f"  Bar: C1 >= {C1_BAR}, C2 >= {C2_BAR}, inconsistency <= {INCONSISTENCY_BAR}")

    corpus_df, test_qdp, _ = load_corpus_and_queries("test")
    targets = build_question_targets(test_qdp)

    judge = LlamaJudge(model_name=args.judge_model,
                       load_in_4bit=not args.no_4bit, seed=args.seed)

    print_section_header("STEP 1 — validation")
    val = run_validation(judge, targets, corpus_df, args.limit, args.seed)
    val.to_csv(OUT_DIR / "judge_validation_raw.csv", index=False)
    vsum = summarise_validation(val)
    print()
    print(vsum.to_string(index=False))
    vsum.to_csv(OUT_DIR / "judge_validation_summary.csv", index=False)

    def acc(c):
        r = vsum[vsum.condition == c]
        return float(r["accuracy"].iloc[0]) if len(r) else 0.0
    c1, c2, c3 = acc("C1_gold_vs_random"), acc("C2_gold_vs_correct"), acc("C3_gold_vs_perturbed")
    inc = float(val[val.condition != "C3_gold_vs_perturbed"]["outcome"]
                .eq("inconsistent").mean()) if len(val) else 1.0

    passed = (c1 >= C1_BAR) and (c2 >= C2_BAR) and (inc <= INCONSISTENCY_BAR)
    print_section_header("GATE")
    print(f"  C1 gold vs random     : {c1:.3f}  (bar {C1_BAR})")
    print(f"  C2 gold vs correct    : {c2:.3f}  (bar {C2_BAR})")
    print(f"  C3 gold vs perturbed  : {c3:.3f}  (probe, not gated — "
          f"automatic metrics score ~0.56 here)")
    print(f"  order-inconsistency   : {inc:.3f}  (bar <= {INCONSISTENCY_BAR})")
    print(f"\n  VERDICT: {'PASS' if passed else 'FAIL'}")

    if not passed and not args.force:
        print("\n  ❌ Judge failed its positive control. NOT scoring the experiment.")
        print("     A judge that cannot distinguish known-good from known-bad")
        print("     distractors cannot support any conclusion — the same reason")
        print("     the Exact Match result was withdrawn.")
        write_report(vsum, None, c1, c2, c3, inc, passed)
        sys.exit(2)

    if args.validate_only:
        write_report(vsum, None, c1, c2, c3, inc, passed)
        print("\n  Validation-only run complete.")
        return

    print_section_header("STEP 2 — experiment")
    arms = load_arms()
    for a in arms:
        print(f"  {a}: {len(arms[a])} questions — {ARM_DESCRIPTIONS.get(a,'')}")
    pairs = [("G5", "G4"), ("G1", "G4"), ("GOLD", "G4"), ("GOLD", "G1")]
    exp = run_experiment(judge, arms, pairs, args.exp_limit)
    if len(exp):
        exp.to_csv(OUT_DIR / "judge_experiment_raw.csv", index=False)
        esum = summarise_experiment(exp)
        print()
        print(esum.to_string(index=False))
        esum.to_csv(OUT_DIR / "judge_experiment_summary.csv", index=False)
    else:
        esum = None

    write_report(vsum, esum, c1, c2, c3, inc, passed)
    print("\n  ✅ LLM-judge evaluation complete.")


def write_report(vsum, esum, c1, c2, c3, inc, passed) -> None:
    def md(f, index=False):
        if f is None:
            return "_not run_"
        try:
            return f.to_markdown(index=index)
        except ImportError:
            return "```\n" + f.to_string(index=index) + "\n```"

    gate = (f"**{'PASS' if passed else 'FAIL'}** — C1 {c1:.3f} (bar {C1_BAR}), "
            f"C2 {c2:.3f} (bar {C2_BAR}), order-inconsistency {inc:.3f} "
            f"(bar <= {INCONSISTENCY_BAR}).")

    probe = (f"On C3 (gold vs digit-perturbed gold) the judge scores {c3:.3f}. "
             + ("This clears the blind spot that defeats every automatic metric "
                "(AUC ~0.56), so the judge adds genuine capability."
                if c3 >= 0.70 else
                "This does NOT clear the automatic-metric blind spot (~0.56), so "
                "'right misconception, wrong arithmetic' remains undetectable by "
                "any instrument in this project."))

    verdict = "_experiment not scored_"
    if esum is not None and len(esum):
        g = esum[esum.comparison == "G4 vs GOLD"]
        parts = []
        if len(g):
            wr = float(g["treatment_win_rate"].iloc[0])
            parts.append(f"Generated distractors are preferred over teacher-written "
                         f"gold ones **{wr:.1%}** of the time.")
        m = esum[esum.comparison == "G4 vs G5"]
        if len(m):
            r = m.iloc[0]
            wr, p = float(r["treatment_win_rate"]), float(r["p_vs_chance"])
            parts.append(
                f"Oracle context beats anti-oracle context {wr:.1%} of the time "
                f"(p={p:.4g}), which is "
                + ("**significant** — misconception-matched context produces "
                   "distractors a judge can tell apart."
                   if p < 0.05 else
                   "**not significant**, consistent with the small bounded effect "
                   "found by the validated automatic metric."))
        verdict = " ".join(parts)

    lines = [
        "# LLM-as-Judge Evaluation", "",
        "Substitute for unavailable human evaluation, covering plausibility and "
        "educational usefulness — constructs no automatic metric measures.", "",
        "**Validity without human labels.** The judge is validated against pairs "
        "whose correct answer is known by construction, the same positive-control "
        "battery that exposed Exact Match as blind (AUC 0.507). Every pair is "
        "judged in both A/B orders so position bias is measured, not assumed away. "
        "The judge is a different model family from the generator, avoiding "
        "self-preference.", "",
        "## Step 1 — validation", "", md(vsum), "",
        f"**Gate:** {gate}", "", probe, "",
        "## Step 2 — experiment", "", md(esum), "",
        "## Verdict", "", verdict, "",
        "## Caveats", "",
        "- The judge is an imperfect proxy for teachers; it validates only that it "
        "separates known-good from known-bad distractors on this data.",
        "- Ties and order-inconsistent verdicts are excluded from win rates and "
        "reported separately; a high inconsistency rate would undermine all "
        "verdicts.",
        "- One candidate per arm per question (the first valid one) is judged, so "
        "results reflect representative rather than best-of-M quality.",
        "", "---", "*Auto-generated by scripts/13_llm_judge.py.*",
    ]
    p = OUT_DIR / "llm_judge_report.md"
    p.write_text("\n".join(lines))
    print(f"  [Saved] {p}")


if __name__ == "__main__":
    main()
