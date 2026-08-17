#!/usr/bin/env python3
"""
Stage 10 — Evaluation for Experiment 2.1 (staged mechanism test).

Reads the per-arm JSONL files written by scripts/09_run_mechanism.py and
produces the per-question table, summary, paired significance tests, figure,
and an auto-generated report.

Primary comparison
    G4 (oracle) vs G5 (anti-oracle) on their common questions, exact-match
    rate. Pre-specified; every other comparison is secondary and
    Holm-corrected within its family.

Usage
    python scripts/10_evaluate_mechanism.py
    python scripts/10_evaluate_mechanism.py --smoke   # evaluate the smoke run
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR, RESULTS_DIR, FIGURES_DIR
from src.utils import print_section_header, setup_matplotlib
from src.gen_evaluation import (
    build_per_question_table, add_semantic_metrics, compare_arms, holm_correct,
)
from src.generation import ARM_DESCRIPTIONS

GEN_DIR = OUTPUT_DIR / "generation" / "exp21"
OUT_DIR = RESULTS_DIR / "exp21"
FIG_DIR = FIGURES_DIR / "exp21"

PRIMARY = ("G5", "G4")           # (reference, treatment)
SECONDARY = [("G1", "G4"), ("G1", "G5")]
METRICS = ["exact_match", "distractor_similarity", "misconception_similarity"]


def load_arms(smoke: bool) -> dict:
    suffix = "_smoke" if smoke else ""
    out = {}
    for path in sorted(GEN_DIR.glob(f"*_seed*{suffix}.jsonl")):
        if not smoke and path.name.endswith("_smoke.jsonl"):
            continue
        arm = path.name.split("_")[0]
        recs = [json.loads(l) for l in open(path) if l.strip()]
        if recs:
            out.setdefault(arm, []).extend(recs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print_section_header("EXPERIMENT 2.1 — EVALUATION")
    records_by_arm = load_arms(args.smoke)
    if not records_by_arm:
        sys.exit(f"No generations found in {GEN_DIR}. Run scripts/09_run_mechanism.py first.")
    for arm, recs in records_by_arm.items():
        print(f"  {arm}: {len(recs)} records — {ARM_DESCRIPTIONS.get(arm, '')}")

    # ---- Per-question metrics ----
    df = build_per_question_table(records_by_arm)
    flat = [r for arm in records_by_arm for r in records_by_arm[arm]]
    order = pd.MultiIndex.from_arrays([df.arm, df.question_id])
    lookup = {(r["arm"], r["question_id"]): r for r in flat}
    ordered = [lookup[(a, q)] for a, q in order]
    df = add_semantic_metrics(df, ordered)
    df.to_csv(OUT_DIR / "exp21_per_question.csv", index=False)
    print(f"\n  [Saved] {OUT_DIR / 'exp21_per_question.csv'}")

    # ---- Manipulation check: did the arms differ as intended? ----
    print_section_header("Manipulation check (matched exemplars per arm)")
    mc = df.groupby("arm").agg(
        n_questions=("question_id", "nunique"),
        mean_matched_exemplars=("n_exemplars_matched", "mean"),
    ).round(3)
    print(mc.to_string())
    mc.to_csv(OUT_DIR / "exp21_manipulation_check.csv")

    # ---- Summary ----
    print_section_header("Summary by arm")
    summary = df.groupby("arm").agg(
        n=("question_id", "nunique"),
        exact_match=("exact_match", "mean"),
        distractor_similarity=("distractor_similarity", "mean"),
        misconception_similarity=("misconception_similarity", "mean"),
        collides_with_correct=("collides_with_correct", "mean"),
        has_duplicates=("has_duplicates", "mean"),
        parse_failed=("parse_failed", "mean"),
    ).round(4)
    print(summary.to_string())
    summary.to_csv(OUT_DIR / "exp21_summary.csv")

    # ---- Paired significance ----
    print_section_header("Paired comparisons")
    arms = set(df.arm)
    blocks = []
    if set(PRIMARY) <= arms:
        prim = compare_arms(df, PRIMARY[0], PRIMARY[1], METRICS)
        prim.insert(0, "family", "PRIMARY")
        blocks.append(prim)
    for a, b in SECONDARY:
        if {a, b} <= arms:
            sec = compare_arms(df, a, b, METRICS)
            if len(sec):
                sec.insert(0, "family", "secondary")
                blocks.append(sec)

    if blocks:
        sig = pd.concat(blocks, ignore_index=True)
        # Holm correction applies within the secondary family only; the single
        # pre-specified primary comparison needs no correction.
        sig["p_holm"] = "-"
        mask = sig.family == "secondary"
        if mask.any():
            adj = holm_correct([float(p) for p in sig.loc[mask, "p_value"]])
            sig.loc[mask, "p_holm"] = [f"{v:.4g}" for v in adj]
        print(sig.to_string(index=False))
        sig.to_csv(OUT_DIR / "exp21_significance.csv", index=False)
    else:
        sig = pd.DataFrame()

    # ---- Figure ----
    if len(summary) > 1:
        show = [m for m in METRICS if summary[m].notna().any()]
        fig, axes = plt.subplots(1, len(show), figsize=(4.6 * len(show), 4.4))
        axes = np.atleast_1d(axes)
        colours = {"G0": "#94a3b8", "G1": "#94a3b8", "G4": "#16a34a", "G5": "#dc2626"}
        for ax, m in zip(axes, show):
            arms_sorted = list(summary.index)
            vals = summary.loc[arms_sorted, m].values
            ax.bar(arms_sorted, vals,
                   color=[colours.get(a, "#2563eb") for a in arms_sorted],
                   edgecolor="white")
            ax.set_title(m.replace("_", " "))
            ax.set_ylabel("mean")
        fig.suptitle("Experiment 2.1 — does misconception-matched context help?\n"
                     "G4 oracle (green) vs G5 anti-oracle (red) vs G1 random (grey)",
                     y=1.04)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "exp21_arms.png", dpi=200, bbox_inches="tight")
        print(f"\n  [Saved] {FIG_DIR / 'exp21_arms.png'}")

    write_report(summary, sig, mc)
    print("\n  ✅ Evaluation complete.")


def write_report(summary: pd.DataFrame, sig: pd.DataFrame, mc: pd.DataFrame) -> None:
    def md(frame, index=True):
        try:
            return frame.to_markdown(index=index)
        except ImportError:
            return "```\n" + frame.to_string(index=index) + "\n```"

    verdict = "Primary comparison unavailable (G4 or G5 missing)."
    if len(sig):
        p = sig[(sig.family == "PRIMARY") & (sig.metric == "exact_match")]
        if len(p):
            row = p.iloc[0]
            d, pv = float(row["delta"]), float(row["p_value"])
            n = int(row["n_paired"])
            if pv < 0.05 and d > 0:
                verdict = (
                    f"**Misconception-matched context causally improves generation.** "
                    f"Oracle exemplars raise exact-match by {d:+.3f} over anti-oracle "
                    f"(n={n} paired, p={pv:.4g}). Retrieval quality therefore has "
                    f"headroom to exploit, and further retrieval work is justified. "
                    f"The delta is an approximate upper bound on what any retriever "
                    f"could deliver at k=5.")
            elif pv < 0.05 and d < 0:
                verdict = (
                    f"**Unexpected: matched context significantly HURTS** "
                    f"({d:+.3f}, p={pv:.4g}, n={n}). This warrants inspection of the "
                    f"prompts before any further retrieval investment.")
            else:
                verdict = (
                    f"**No detectable effect of misconception-matched context** "
                    f"({d:+.3f}, p={pv:.4g}, n={n}). Even a perfect retriever would "
                    f"not measurably improve generation at k=5 under this prompt. "
                    f"This undercuts the premise that better retrieval improves "
                    f"generation, and means the G2/G3 comparison — whose upstream "
                    f"contrast is far smaller — is not worth running as specified. "
                    f"Redirect effort to the prompt or the generator instead.")

    lines = [
        "# Experiment 2.1 — Mechanism Test", "",
        "**Question.** Does misconception-matched context causally improve generated "
        "distractors? This is the precondition for retrieval quality to matter at all.",
        "",
        "**Design.** Exemplar selection is the only variable; prompt template, "
        "generator, decoding parameters, k, and evaluation set are fixed. "
        "G4 (oracle) and G5 (anti-oracle) are diagnostics that consult gold labels, "
        "not deployable systems. G5 draws from the same subject as the target, so the "
        "G4/G5 contrast isolates misconception match rather than topical similarity.",
        "",
        "## Manipulation check", "", md(mc), "",
        "## Results by arm", "", md(summary), "",
        "## Paired comparisons", "",
        md(sig, index=False) if len(sig) else "_none_", "",
        "## Verdict", "", verdict, "",
        "## Interpretation notes", "",
        "- Semantic metrics use the PRETRAINED encoder, never the fine-tuned "
        "retriever, to avoid favouring arms whose exemplars that model selected.",
        "- `misconception_similarity` compares the model's SELF-REPORTED intent "
        "against the gold label; it does not verify what the distractor actually "
        "encodes.",
        "- Exact-match is binary and low-powered at this sample size; the continuous "
        "metrics are more sensitive and should be weighted accordingly when the two "
        "disagree.",
        "- G4 is constructible only where the corpus holds >= k QDPs sharing a gold "
        "misconception (62% of test questions), so the primary contrast runs on the "
        "common subset and is not representative of the full long tail.",
        "", "---", "*Auto-generated by scripts/10_evaluate_mechanism.py.*",
    ]
    path = OUT_DIR / "exp21_report.md"
    path.write_text("\n".join(lines))
    print(f"  [Saved] {path}")


if __name__ == "__main__":
    main()
