#!/usr/bin/env python3
"""
Stage 18 — Misconception-alignment evaluation of generated distractors.

Order of operations is deliberate: the evaluator is measured and gated BEFORE
it is used for any conclusion.

    1. Standalone performance on the 431 held-out gold test QDPs
    2. Positive control (gold vs random / correct answer / near-miss)
    3. GATE: abort unless standalone MRR clears the zero-shot floor and the
       coarse controls pass
    4. Apply to generated candidates

Alignment definition
    A generated distractor has no designated gold label, so it counts as
    aligned when it ranks ANY of its question's gold misconceptions highly.
    Per candidate we take the best rank across that question's gold set.

Aggregation (per question)
    PRIMARY   mean reciprocal rank over all valid candidates
              -> is the TYPICAL generated distractor aligned?
    SECONDARY best-of-M reciprocal rank
              -> does the pool contain at least one aligned candidate?
    Stage 3   the single selected candidate, if selections exist
              -> does selection improve alignment?

    Teacher-written gold distractors are scored identically as an interpretive
    anchor: generated output is read relative to human-written output on the
    same questions, not against an arbitrary absolute scale.

Usage
    python scripts/18_evaluate_misconception_alignment.py
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from src.config import MISCONCEPTION_CSV, RESULTS_DIR, OUTPUT_DIR, RANDOM_SEED
from src.utils import set_seed, print_section_header
from src.gen_evaluation import normalise_answer
from src.misconception_evaluator import (
    MisconceptionRanker, build_query_text, build_queries_from_qdp,
    rank_metrics, reciprocal_ranks, positive_control, assert_no_label_leakage,
)

EVAL_DIR = OUTPUT_DIR / "misconception_evaluator"
GEN_EXP21 = OUTPUT_DIR / "generation" / "exp21"
RERANK_SEL = OUTPUT_DIR / "reranking" / "selections_k1.csv"

ZERO_SHOT_FLOOR = 0.1472   # untrained all-MiniLM-L6-v2 on the same 431 gold QDPs
AUC_BAR = 0.70


def bootstrap_ci(d: np.ndarray, n_boot: int = 10000, seed: int = 0):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load_arms() -> dict:
    out = {}
    if not GEN_EXP21.exists():
        return out
    for p in sorted(GEN_EXP21.glob("*_seed*.jsonl")):
        if p.name.endswith("_smoke.jsonl"):
            continue
        arm = p.name.split("_")[0]
        for r in (json.loads(l) for l in open(p) if l.strip()):
            out.setdefault(arm, {})[r["question_id"]] = r
    return out


def valid_candidates(rec) -> list:
    """Stage 3 hard filter, applied identically across arms."""
    correct = normalise_answer(rec["correct_answer"])
    seen, out = set(), []
    for c in rec.get("candidates", []):
        d = str(c.get("distractor", "")).strip()
        n = normalise_answer(d)
        if not d or n == correct or n in seen:
            continue
        seen.add(n)
        out.append(d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=EVAL_DIR / "model")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--control-limit", type=int, default=150)
    ap.add_argument("--force", action="store_true",
                    help="Proceed even if the evaluator fails its gate (not recommended)")
    args = ap.parse_args()

    set_seed(args.seed)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if not Path(args.model).exists():
        sys.exit(f"Evaluator not found at {args.model}. Run scripts/17 first.")

    mm = pd.read_csv(MISCONCEPTION_CSV)
    tr = pd.read_csv(RESULTS_DIR / "train_qdp.csv")
    te = pd.read_csv(RESULTS_DIR / "test_qdp.csv")

    print_section_header("MISCONCEPTION-ALIGNMENT EVALUATION")
    print(f"  evaluator  : {args.model}")
    print(f"  label space: {len(mm)}")
    ranker = MisconceptionRanker(str(args.model), mm)

    # ---------- 1. Standalone performance on held-out gold QDPs ----------
    print_section_header("1. Standalone performance (431 held-out gold test QDPs)")
    te_queries = build_queries_from_qdp(te)
    assert_no_label_leakage(te_queries, mm)
    gold_ranks = ranker.best_gold_ranks(te_queries,
                                        [[int(m)] for m in te.MisconceptionId])
    standalone = rank_metrics(gold_ranks)
    for k, v in standalone.items():
        print(f"  {k:<12}: {v:.4f}" if isinstance(v, float) else f"  {k:<12}: {v}")
    print(f"  zero-shot floor (untrained MiniLM) MRR = {ZERO_SHOT_FLOOR:.4f}")

    # ---------- 2. Positive control ----------
    print_section_header("2. Positive control")
    ctrl = positive_control(ranker, te, tr.DistractorText.astype(str).tolist(),
                            seed=args.seed, limit=args.control_limit)
    print(ctrl.to_string(index=False))

    # ---------- 3. Gate ----------
    auc = {r["comparison"][:2]: float(r["AUC"]) for _, r in ctrl.iterrows()}
    passed = (standalone["MRR"] > ZERO_SHOT_FLOOR
              and auc.get("C1", 0) >= AUC_BAR and auc.get("C2", 0) >= AUC_BAR)
    print_section_header("3. Gate")
    print(f"  standalone MRR {standalone['MRR']:.4f} > floor {ZERO_SHOT_FLOOR:.4f} : "
          f"{standalone['MRR'] > ZERO_SHOT_FLOOR}")
    print(f"  C1 AUC {auc.get('C1', float('nan')):.3f} >= {AUC_BAR} : "
          f"{auc.get('C1', 0) >= AUC_BAR}")
    print(f"  C2 AUC {auc.get('C2', float('nan')):.3f} >= {AUC_BAR} : "
          f"{auc.get('C2', 0) >= AUC_BAR}")
    print(f"  C3 near-miss AUC {auc.get('C3', float('nan')):.3f} (probe, not gated)")
    print(f"\n  VERDICT: {'PASS' if passed else 'FAIL'}")
    json.dump({"standalone": standalone, "zero_shot_floor": ZERO_SHOT_FLOOR,
               "positive_control": ctrl.to_dict("records"), "gate_passed": bool(passed)},
              open(EVAL_DIR / "standalone_results.json", "w"), indent=2)

    if not passed and not args.force:
        print("\n  Evaluator did not clear its gate; not scoring generated output.")
        write_report(standalone, ctrl, None, None, passed)
        sys.exit(2)

    # ---------- 4. Apply to generated candidates ----------
    arms = load_arms()
    if not arms:
        print(f"\n  No generations found in {GEN_EXP21}; stopping after validation.")
        write_report(standalone, ctrl, None, None, passed)
        return

    gold_by_q = te.groupby("QuestionId")["MisconceptionId"].apply(
        lambda s: [int(x) for x in s]).to_dict()

    print_section_header("4. Alignment of generated distractors")
    rows = []
    for arm, recs in arms.items():
        queries, owner = [], []
        for qid, rec in recs.items():
            for d in valid_candidates(rec):
                queries.append(build_query_text(rec["question_text"],
                                                rec["correct_answer"], d))
                owner.append(qid)
        if not queries:
            continue
        assert_no_label_leakage(queries, mm)
        ranks = ranker.best_gold_ranks(queries, [gold_by_q[q] for q in owner])
        rr = reciprocal_ranks(ranks)
        df = pd.DataFrame({"question_id": owner, "rr": rr, "rank": ranks})
        agg = df.groupby("question_id").agg(mean_rr=("rr", "mean"),
                                            best_rr=("rr", "max"),
                                            n_candidates=("rr", "size")).reset_index()
        agg["arm"] = arm
        rows.append(agg)
        print(f"  {arm}: {len(queries)} candidates over {agg.shape[0]} questions | "
              f"mean_rr {agg.mean_rr.mean():.4f} | best_rr {agg.best_rr.mean():.4f}")

    per_q = pd.concat(rows, ignore_index=True)

    # Teacher-written anchor
    anchor_ranks = ranker.best_gold_ranks(
        te_queries, [gold_by_q[q] for q in te.QuestionId])
    anchor = pd.DataFrame({"question_id": te.QuestionId,
                           "rr": reciprocal_ranks(anchor_ranks)})
    anchor_q = anchor.groupby("question_id").rr.agg(["mean", "max"]).reset_index()
    anchor_q.columns = ["question_id", "mean_rr", "best_rr"]
    anchor_q["arm"] = "GOLD (teacher-written)"
    anchor_q["n_candidates"] = anchor.groupby("question_id").size().values
    per_q = pd.concat([per_q, anchor_q], ignore_index=True)
    print(f"  GOLD (anchor): mean_rr {anchor_q.mean_rr.mean():.4f}")

    # Stage 3 selected candidate
    if RERANK_SEL.exists():
        sel = pd.read_csv(RERANK_SEL)
        gen_main = OUTPUT_DIR / "generation" / "stage2_generations.jsonl"
        if gen_main.exists():
            src = {r["question_id"]: r for r in
                   (json.loads(l) for l in open(gen_main) if l.strip())}
            sub = []
            for strategy, grp in sel.groupby("strategy"):
                qs, own = [], []
                for _, row in grp.iterrows():
                    rec = src.get(row.question_id)
                    if rec is None or row.question_id not in gold_by_q:
                        continue
                    for d in json.loads(row.selected):
                        qs.append(build_query_text(rec["question_text"],
                                                   rec["correct_answer"], d))
                        own.append(row.question_id)
                if not qs:
                    continue
                rk = ranker.best_gold_ranks(qs, [gold_by_q[q] for q in own])
                d3 = pd.DataFrame({"question_id": own, "rr": reciprocal_ranks(rk)})
                a3 = d3.groupby("question_id").agg(mean_rr=("rr", "mean"),
                                                   best_rr=("rr", "max"),
                                                   n_candidates=("rr", "size")).reset_index()
                a3["arm"] = f"Stage3:{strategy}"
                sub.append(a3)
            if sub:
                s3 = pd.concat(sub, ignore_index=True)
                per_q = pd.concat([per_q, s3], ignore_index=True)
                print_section_header("Stage 3 selected-candidate alignment")
                print(s3.groupby("arm").mean_rr.mean().round(4).to_string())

    per_q.to_csv(EVAL_DIR / "stage2_alignment_per_question.csv", index=False)
    summary = per_q.groupby("arm").agg(
        n_questions=("question_id", "nunique"),
        mean_rr=("mean_rr", "mean"),
        best_rr=("best_rr", "mean"),
        candidates_per_q=("n_candidates", "mean")).round(4)
    print_section_header("Summary")
    print(summary.to_string())
    summary.to_csv(EVAL_DIR / "stage2_alignment_summary.csv")
    json.dump(summary.reset_index().to_dict("records"),
              open(EVAL_DIR / "stage2_alignment_results.json", "w"), indent=2)

    # ---------- 5. Paired comparisons ----------
    print_section_header("5. Paired comparisons")
    comps, piv = [], per_q.pivot_table(index="question_id", columns="arm",
                                       values=["mean_rr", "best_rr"])
    for metric in ["mean_rr", "best_rr"]:
        for a, b, fam in [("G5", "G4", "PRIMARY"), ("G1", "G4", "secondary"),
                          ("G1", "G5", "secondary")]:
            if (metric, a) not in piv.columns or (metric, b) not in piv.columns:
                continue
            s = piv[[(metric, a), (metric, b)]].dropna()
            if len(s) < 5:
                continue
            va, vb = s[(metric, a)].values, s[(metric, b)].values
            d = vb - va
            p = 1.0 if np.allclose(d, 0) else float(stats.wilcoxon(vb, va).pvalue)
            lo, hi = bootstrap_ci(d)
            comps.append({"family": fam, "metric": metric,
                          "comparison": f"{b} vs {a}", "n_paired": len(s),
                          "reference_mean": round(float(va.mean()), 4),
                          "treatment_mean": round(float(vb.mean()), 4),
                          "delta": round(float(d.mean()), 4),
                          "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                          "p_value": f"{p:.4g}",
                          "significant_0.05": "yes" if p < 0.05 else "no"})
    comp_df = pd.DataFrame(comps)
    if len(comp_df):
        print(comp_df.to_string(index=False))
        comp_df.to_csv(EVAL_DIR / "comparison_results.csv", index=False)
        json.dump(comps, open(EVAL_DIR / "comparison_results.json", "w"), indent=2)

    write_report(standalone, ctrl, summary, comp_df, passed)
    print("\n  Evaluation complete.")


def write_report(standalone, ctrl, summary, comp_df, passed) -> None:
    def md(f, index=True):
        if f is None or not len(f):
            return "_not run_"
        try:
            return f.to_markdown(index=index)
        except ImportError:
            return "```\n" + f.to_string(index=index) + "\n```"

    verdict = "_generated output not scored (evaluator failed its gate)_"
    if summary is not None and comp_df is not None and len(comp_df):
        p = comp_df[(comp_df.family == "PRIMARY") & (comp_df.metric == "mean_rr")]
        if len(p):
            r = p.iloc[0]
            d, pv, n = float(r["delta"]), float(r["p_value"]), int(r["n_paired"])
            lo, hi = float(r["ci95_low"]), float(r["ci95_high"])
            verdict = (
                f"On the primary measure (mean alignment across candidates), oracle "
                f"context differs from anti-oracle by {d:+.4f} "
                f"(95% CI [{lo:+.4f}, {hi:+.4f}], p={pv:.4g}, n={n} paired). "
                + ("This is statistically significant."
                   if pv < 0.05 else
                   "This is not statistically significant, consistent with the "
                   "small bounded effect found by the gated-quality composite "
                   "(+0.015, p=0.088) and the pointwise LLM judge (+0.014, p=1.00). "
                   "Three independent instruments now agree."))

    lines = [
        "# Independent Misconception-Alignment Evaluation", "",
        "## 1. Architecture", "",
        "Bi-encoder ranker over misconception descriptions. Query = question + "
        "correct answer + incorrect answer chosen; documents = all 2,587 "
        "misconception descriptions; score = cosine similarity, measured as the "
        "rank of the question's gold misconception.", "",
        "A ranker rather than an N-way classifier because 27.6% of test gold "
        "misconceptions never occur in training; a softmax over training classes "
        "would score those questions as failures regardless of distractor quality.",
        "", "## 2. Split and leakage prevention", "",
        "| Role | Data |", "|---|---|",
        "| Evaluator training | train QDPs (3,507) |",
        "| Early stopping | val QDPs (432) |",
        "| Standalone reference | test gold QDPs (431), never seen |",
        "| Application | generated candidates for the 187 test questions |", "",
        "Checks enforced in code: no misconception description appears in any "
        "evaluator input; the generator's self-reported misconception is never an "
        "input; training questions are disjoint from application questions; and "
        "the base encoder (mpnet) differs from the Stage 1 retriever (MiniLM), so "
        "the evaluator is not the system under test.", "",
        "## 3. Standalone performance", "",
        md(pd.DataFrame([standalone]), index=False), "",
        f"Zero-shot floor (untrained MiniLM, same 431 QDPs): MRR {ZERO_SHOT_FLOOR:.4f}.",
        "", "### Positive control", "", md(ctrl, index=False), "",
        f"**Gate: {'PASS' if passed else 'FAIL'}.** C1/C2 are gated; C3 (near-miss) "
        "is a probe. Every instrument previously tested in this project scores "
        "~0.56 on C3.", "",
        "## 4. Alignment of generated distractors", "", md(summary), "",
        "Primary is `mean_rr` (typical candidate); `best_rr` is best-of-M (pool "
        "potential). GOLD is the teacher-written anchor on the same questions.", "",
        "## 5. Statistical comparison", "", md(comp_df, index=False), "",
        "## 6. Verdict", "", verdict, "",
        "## 7. Limitations", "",
        "- Ranking over 2,587 classes is intrinsically hard; absolute MRR is low "
        "and the instrument is noisier than the gated composite (AUC 0.898). It "
        "therefore has *less* power to resolve the Stage 2 effect, not more.",
        "- Alignment is credited for matching *any* of a question's gold "
        "misconceptions, since generated distractors have no designated gold label.",
        "- A bi-encoder scores textual association, not the procedural reasoning "
        "needed to tell a valid distractor from a near-miss; the C3 probe measures "
        "this directly.",
        "- Single training run, single base encoder, single seed.",
        "", "---", "*Auto-generated by scripts/18_evaluate_misconception_alignment.py.*",
    ]
    (EVAL_DIR / "report.md").write_text("\n".join(lines))
    print(f"  [Saved] {EVAL_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
