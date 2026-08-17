#!/usr/bin/env python3
"""
Stage 12 — Re-analysis of Experiment 2.1 with a validated metric.

Why this exists
    The Experiment 2.1 headline used Exact Match, which subsequently failed
    its positive control: it credits a known-good distractor only 1.8% of the
    time and separates good from random at AUC 0.507 (chance = 0.500). A null
    from a blind instrument carries no information, so that conclusion was
    withdrawn.

    This re-analyses the SAME generations (no new generation) using a metric
    that is validated first, in this script, before it is used.

Procedure
    STEP 1  Validate the gated-quality composite on the positive control
            (T1-T5 conditions with known quality). The composite is used in
            step 2 only if it clears the pre-set bar.
    STEP 2  Re-analyse G1/G4/G5 with the validated composite as the
            pre-specified primary; Exact Match is demoted to a reported floor.

Pre-specified bar for step 1
    AUC(T2 good vs T3 random) >= 0.70  AND  AUC(T2 good vs T4 correct answer)
    >= 0.70. The second condition is what raw embedding similarity fails
    (AUC 0.425) and is the reason the gate exists.

Usage
    python scripts/12_reanalyse_exp21.py
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

from src.config import RESULTS_DIR, OUTPUT_DIR, RANDOM_SEED
from src.utils import set_seed, print_section_header
from src.retrieval_experiment import load_corpus_and_queries
from src.generation import build_question_targets, ARM_DESCRIPTIONS
from src.gen_evaluation import (
    normalise_answer, build_per_question_table, add_gated_quality,
    compare_arms, holm_correct,
)

GEN_DIR = OUTPUT_DIR / "generation" / "exp21"
OUT_DIR = RESULTS_DIR / "exp21_reanalysis"

AUC_BAR = 0.70
PRIMARY_METRIC = "gated_quality_max"
METRICS = ["gated_quality_max", "gated_quality_mean", "exact_match"]


def _auc(good: np.ndarray, bad: np.ndarray):
    u, p = stats.mannwhitneyu(good, bad, alternative="two-sided")
    return float(u / (len(good) * len(bad))), float(p)


def validate_composite(seed: int) -> pd.DataFrame:
    """
    STEP 1 — positive control for the gated composite.

    Builds single-candidate pseudo-records for each known-quality condition
    and scores them with exactly the function used in step 2, so the metric
    validated is the metric applied.
    """
    rng = np.random.RandomState(seed)
    corpus_df, test_qdp, _ = load_corpus_and_queries("test")
    targets = build_question_targets(test_qdp)
    multi = [t for t in targets if len(t["gold_distractors"]) >= 2]
    corpus_distractors = corpus_df["DistractorText"].tolist()

    records, conds = [], []
    for t in multi:
        golds = t["gold_distractors"]
        for i, gi in enumerate(golds):
            others = [g for j, g in enumerate(golds) if j != i]
            rand_d = corpus_distractors[rng.randint(len(corpus_distractors))]
            for cond, cand in [
                ("T2_good_not_in_ref", gi["distractor"]),
                ("T3_random_distractor", rand_d),
                ("T4_correct_answer", t["correct_answer"]),
            ]:
                records.append({
                    "question_id": t["question_id"],
                    "correct_answer": t["correct_answer"],
                    "gold_distractors": others,
                    "candidates": [{"distractor": str(cand), "stated_misconception": ""}],
                })
                conds.append(cond)

    df = pd.DataFrame({"condition": conds})
    df = add_gated_quality(df, records)

    rows = []
    good = df[df.condition == "T2_good_not_in_ref"][PRIMARY_METRIC].values
    for bad_cond, label in [("T3_random_distractor", "T2 vs T3 (good vs random)"),
                            ("T4_correct_answer", "T2 vs T4 (good vs correct answer)")]:
        bad = df[df.condition == bad_cond][PRIMARY_METRIC].values
        auc, p = _auc(good, bad)
        rows.append({"comparison": label, "good_mean": round(float(good.mean()), 4),
                     "bad_mean": round(float(bad.mean()), 4),
                     "AUC": round(auc, 4), "p_value": f"{p:.3g}",
                     "passes_bar": "YES" if auc >= AUC_BAR else "NO"})
    return pd.DataFrame(rows)


def load_arms() -> dict:
    out = {}
    for path in sorted(GEN_DIR.glob("*_seed*.jsonl")):
        if path.name.endswith("_smoke.jsonl"):
            continue
        arm = path.name.split("_")[0]
        recs = [json.loads(l) for l in open(path) if l.strip()]
        if recs:
            out.setdefault(arm, []).extend(recs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()
    set_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- STEP 1 ----------------
    print_section_header("STEP 1 — validate the gated composite")
    print(f"  Pre-specified bar: AUC >= {AUC_BAR} on BOTH comparisons.")
    val = validate_composite(args.seed)
    print()
    print(val.to_string(index=False))
    val.to_csv(OUT_DIR / "composite_validation.csv", index=False)

    passed = (val["AUC"] >= AUC_BAR).all()
    if not passed:
        print("\n  ❌ Composite FAILED its positive control. Aborting: using it "
              "would repeat the Exact Match error.")
        sys.exit(1)
    print("\n  ✅ Composite passes. Proceeding to re-analysis.")

    # ---------------- STEP 2 ----------------
    print_section_header("STEP 2 — re-analyse Experiment 2.1")
    arms = load_arms()
    if not arms:
        sys.exit(f"No generations in {GEN_DIR}.")
    for a, r in arms.items():
        print(f"  {a}: {len(r)} records — {ARM_DESCRIPTIONS.get(a,'')}")

    df = build_per_question_table(arms)
    lookup = {(r["arm"], r["question_id"]): r for arm in arms for r in arms[arm]}
    ordered = [lookup[(a, q)] for a, q in zip(df.arm, df.question_id)]
    df = add_gated_quality(df, ordered)
    df.to_csv(OUT_DIR / "reanalysis_per_question.csv", index=False)

    print_section_header("Summary by arm")
    summary = df.groupby("arm").agg(
        n=("question_id", "nunique"),
        gated_quality_max=("gated_quality_max", "mean"),
        gated_quality_mean=("gated_quality_mean", "mean"),
        exact_match=("exact_match", "mean"),
        collides_with_correct=("collides_with_correct", "mean"),
        has_duplicates=("has_duplicates", "mean"),
    ).round(4)
    print(summary.to_string())
    summary.to_csv(OUT_DIR / "reanalysis_summary.csv")

    print_section_header("Paired comparisons (primary = gated_quality_max)")
    blocks = []
    if {"G4", "G5"} <= set(df.arm):
        b = compare_arms(df, "G5", "G4", METRICS); b.insert(0, "family", "PRIMARY"); blocks.append(b)
    for a, t in [("G1", "G4"), ("G1", "G5")]:
        if {a, t} <= set(df.arm):
            b = compare_arms(df, a, t, METRICS)
            if len(b):
                b.insert(0, "family", "secondary"); blocks.append(b)
    sig = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
    if len(sig):
        sig["p_holm"] = "-"
        m = sig.family == "secondary"
        if m.any():
            sig.loc[m, "p_holm"] = [f"{v:.4g}" for v in
                                    holm_correct([float(p) for p in sig.loc[m, "p_value"]])]
        print(sig.to_string(index=False))
        sig.to_csv(OUT_DIR / "reanalysis_significance.csv", index=False)

    write_report(val, summary, sig)
    print("\n  ✅ Re-analysis complete.")


def write_report(val, summary, sig) -> None:
    def md(f, index=True):
        try:
            return f.to_markdown(index=index)
        except ImportError:
            return "```\n" + f.to_string(index=index) + "\n```"

    verdict = "Primary comparison unavailable."
    if len(sig):
        p = sig[(sig.family == "PRIMARY") & (sig.metric == PRIMARY_METRIC)]
        if len(p):
            r = p.iloc[0]
            d, pv, n = float(r["delta"]), float(r["p_value"]), int(r["n_paired"])
            lo, hi = float(r["ci95_low"]), float(r["ci95_high"])
            # positive-control scale: good ~0.70, random ~0.29 -> range ~0.41
            pct = d / 0.41 * 100
            if pv < 0.05 and d > 0:
                verdict = (
                    f"**Misconception-matched context significantly improves distractor "
                    f"quality** on a validated metric: {d:+.4f} (95% CI [{lo:+.4f}, "
                    f"{hi:+.4f}], p={pv:.4g}, n={n} paired). On the positive-control "
                    f"scale (good ~0.70, random ~0.29) this is ~{pct:.0f}% of the "
                    f"good-to-bad range — a real but modest effect. Retrieval quality "
                    f"therefore has genuine, bounded headroom.")
            else:
                verdict = (
                    f"**No statistically significant effect** on the validated metric: "
                    f"{d:+.4f} (95% CI [{lo:+.4f}, {hi:+.4f}], p={pv:.4g}, n={n}). "
                    f"Unlike the withdrawn Exact Match result, this null comes from an "
                    f"instrument shown to discriminate good from bad distractors, so it "
                    f"is informative: the CI bounds any true effect at "
                    f"{hi/0.41*100:.0f}% of the good-to-bad range. Even a perfect "
                    f"retriever would deliver at most a modest gain at k=5.")

    lines = [
        "# Experiment 2.1 — Re-analysis with a Validated Metric", "",
        "The original headline used Exact Match, which failed its positive control "
        "(credits a known-good distractor 1.8% of the time; good-vs-random AUC 0.507 "
        "against a chance level of 0.500). That conclusion was withdrawn. This "
        "re-analyses the **same generations** with a metric validated beforehand.",
        "", "## Step 1 — composite validation (positive control)", "",
        "`gated_quality` = embedding similarity to the nearest gold distractor, with "
        "any candidate equal to the correct answer (or empty) zeroed. The gate exists "
        "because raw similarity rates the correct answer *above* a genuine distractor "
        "(0.754 vs 0.703, AUC 0.425).", "",
        md(val, index=False), "",
        f"Pre-specified bar: AUC >= {AUC_BAR} on both comparisons.", "",
        "## Step 2 — results by arm", "", md(summary), "",
        "## Paired comparisons", "",
        md(sig, index=False) if len(sig) else "_none_", "",
        "## Verdict", "", verdict, "",
        "## Caveats", "",
        "- Exact Match is retained in the tables as a **floor only**; it is known "
        "blind (AUC 0.507) and no conclusion rests on it.",
        "- No metric yet separates a good distractor from a near-miss "
        "(perturbed-gold AUC ~0.56), so 'right misconception, wrong arithmetic' "
        "remains invisible.",
        "- The primary contrast runs on the subset where both arms are constructible "
        "at k=5, over-representing commoner misconceptions.",
        "- Single generation seed; `gated_quality_mean` additionally reflects set "
        "degeneracy (duplicates, collisions), which the max variant ignores.",
        "", "---", "*Auto-generated by scripts/12_reanalyse_exp21.py.*",
    ]
    p = OUT_DIR / "exp21_reanalysis_report.md"
    p.write_text("\n".join(lines))
    print(f"  [Saved] {p}")


if __name__ == "__main__":
    main()
