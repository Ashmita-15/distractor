#!/usr/bin/env python3
"""
Stage 15 — Evaluation for Stage 3, Experiment 3.1.

Pre-specified primary comparison
    R2_heuristic vs R0_first, on the VALIDATED gated-quality composite
    (positive control: AUC 0.898 good-vs-random, 0.995 good-vs-correct-answer).
    Paired across the same questions, Wilcoxon signed-rank plus a bootstrap CI.

Secondary (Holm-corrected within family)
    R1_random vs R0   does any selection beat the model's own ordering?
    R4_oracle vs R0   how much headroom exists in total?
    R4_oracle vs R2   how much headroom the heuristic leaves unexploited.

Metric roles
    gated_quality   PRIMARY. Validated; the only metric conclusions rest on.
    exact_match     SECONDARY / floor. Known blind at the candidate level
                    (AUC 0.507), so it is reported but never arbitrates.
    validity        Descriptive: what the hard filter removed.

No leakage: gold distractors are used only by R4 (a diagnostic ceiling) and
by the evaluation metrics themselves. The Stage 1 retriever is not involved.

Usage
    python scripts/15_evaluate_rerank.py
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
from scipy import stats

from src.config import OUTPUT_DIR, RESULTS_DIR, FIGURES_DIR, RANDOM_SEED
from src.utils import set_seed, print_section_header, setup_matplotlib
from src.gen_evaluation import add_gated_quality, normalise_answer, holm_correct
from src.reranking import STRATEGIES, STRATEGY_R3, STRATEGY_DESCRIPTIONS

GEN_INPUT = OUTPUT_DIR / "generation" / "stage2_generations.jsonl"
RERANK_DIR = OUTPUT_DIR / "reranking"
OUT_DIR = RESULTS_DIR / "stage3"
FIG_DIR = FIGURES_DIR / "stage3"

# Experiment 3.1 primary was R2 vs R0. When judge scores are present
# (Experiment 3.2) the pre-specified primary becomes R3 vs R0.
PRIMARY_31 = ("R0_first", "R2_heuristic")
PRIMARY_32 = ("R0_first", STRATEGY_R3)
SECONDARY_31 = [("R0_first", "R1_random"), ("R0_first", "R4_oracle"),
                ("R2_heuristic", "R4_oracle")]
SECONDARY_32 = [("R0_first", "R1_random"), ("R0_first", "R2_heuristic"),
                ("R0_first", "R4_oracle"), (STRATEGY_R3, "R4_oracle")]


def bootstrap_ci(diff: np.ndarray, n_boot: int = 10000, seed: int = 0):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(df: pd.DataFrame, a: str, b: str) -> list:
    """Paired comparison of strategy b against a, on common questions."""
    A = df[df.strategy == a].set_index("question_id")
    B = df[df.strategy == b].set_index("question_id")
    common = sorted(set(A.index) & set(B.index))
    if not common:
        return []
    A, B = A.loc[common], B.loc[common]
    rows = []

    # PRIMARY metric — continuous, validated
    va, vb = A["gated_quality"].values, B["gated_quality"].values
    d = vb - va
    p = 1.0 if np.allclose(d, 0) else float(stats.wilcoxon(vb, va).pvalue)
    lo, hi = bootstrap_ci(d)
    rows.append({"comparison": f"{b} vs {a}", "metric": "gated_quality",
                 "n_paired": len(common),
                 "reference_mean": round(float(va.mean()), 4),
                 "treatment_mean": round(float(vb.mean()), 4),
                 "delta": round(float(d.mean()), 4),
                 "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                 "test": "Wilcoxon signed-rank", "p_value": f"{p:.4g}",
                 "significant_0.05": "yes" if p < 0.05 else "no"})

    # SECONDARY metric — binary, known blind; McNemar
    ea, eb = A["exact_match"].astype(int).values, B["exact_match"].astype(int).values
    n_a = int(((ea == 1) & (eb == 0)).sum())
    n_b = int(((ea == 0) & (eb == 1)).sum())
    disc = n_a + n_b
    pe = stats.binomtest(n_b, disc, 0.5).pvalue if disc else 1.0
    lo, hi = bootstrap_ci((eb - ea).astype(float))
    rows.append({"comparison": f"{b} vs {a}", "metric": "exact_match (floor)",
                 "n_paired": len(common),
                 "reference_mean": round(float(ea.mean()), 4),
                 "treatment_mean": round(float(eb.mean()), 4),
                 "delta": round(float(eb.mean() - ea.mean()), 4),
                 "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                 "test": f"McNemar exact (discordant={disc})",
                 "p_value": f"{pe:.4g}",
                 "significant_0.05": "yes" if pe < 0.05 else "no"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    set_seed(args.seed)
    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print_section_header("STAGE 3 — EVALUATION (Experiment 3.1)")
    sel_path = RERANK_DIR / f"selections_k{args.k}.csv"
    if not sel_path.exists():
        sys.exit(f"Selections not found: {sel_path}. Run scripts/14_rerank.py first.")
    sel = pd.read_csv(sel_path)
    gen = {r["question_id"]: r
           for r in (json.loads(l) for l in open(GEN_INPUT) if l.strip())}
    print(f"  selections: {len(sel)} rows across {sel.strategy.nunique()} strategies")

    # ---- Score each selection with the validated composite ----
    # Each row becomes a pseudo-record holding ONLY its selected candidate, so
    # add_gated_quality's max reduces to that candidate's own score.
    print_section_header("Scoring selections (validated gated-quality composite)")
    pseudo, keys = [], []
    for _, row in sel.iterrows():
        src = gen[row.question_id]
        chosen = json.loads(row.selected)
        pseudo.append({
            "question_id": row.question_id,
            "correct_answer": src["correct_answer"],
            "gold_distractors": src["gold_distractors"],
            "candidates": [{"distractor": c, "stated_misconception": ""} for c in chosen],
        })
        keys.append((row.strategy, row.question_id))

    scored = add_gated_quality(pd.DataFrame({"i": range(len(pseudo))}), pseudo)
    df = pd.DataFrame({
        "strategy": [k[0] for k in keys],
        "question_id": [k[1] for k in keys],
        "gated_quality": scored["gated_quality_max"].values,
    })

    # exact match of the SELECTED candidate(s) only
    em = []
    for (strategy, qid), rec in zip(keys, pseudo):
        gold = {normalise_answer(g["distractor"]) for g in rec["gold_distractors"]}
        gold.discard("")
        em.append(any(normalise_answer(c["distractor"]) in gold
                      for c in rec["candidates"]))
    df["exact_match"] = em
    df.to_csv(OUT_DIR / "stage3_per_question.csv", index=False)

    print_section_header("Summary by strategy")
    summary = df.groupby("strategy").agg(
        n=("question_id", "nunique"),
        gated_quality=("gated_quality", "mean"),
        exact_match=("exact_match", "mean"),
    ).round(4)
    summary["description"] = [STRATEGY_DESCRIPTIONS.get(s, "") for s in summary.index]
    print(summary.to_string())
    summary.to_csv(OUT_DIR / "stage3_summary.csv")

    # ---- Paired comparisons ----
    # The pre-specified primary depends on which experiment ran: R2 vs R0 for
    # 3.1, R3 vs R0 once judge scores are available (3.2).
    has_r3 = STRATEGY_R3 in set(df.strategy)
    primary = PRIMARY_32 if has_r3 else PRIMARY_31
    secondary = SECONDARY_32 if has_r3 else SECONDARY_31
    print_section_header(
        f"Paired comparisons (primary = {primary[1]} vs {primary[0]})")
    blocks = []
    prim = compare(df, *primary)
    for r in prim:
        r["family"] = "PRIMARY"
    blocks.extend(prim)
    for a, b in secondary:
        sec = compare(df, a, b)
        for r in sec:
            r["family"] = "secondary"
        blocks.extend(sec)

    sig = pd.DataFrame(blocks)
    if len(sig):
        cols = ["family", "comparison", "metric", "n_paired", "reference_mean",
                "treatment_mean", "delta", "ci95_low", "ci95_high", "test",
                "p_value", "significant_0.05"]
        sig = sig[cols]
        sig["p_holm"] = "-"
        m = (sig.family == "secondary") & (sig.metric == "gated_quality")
        if m.any():
            sig.loc[m, "p_holm"] = [f"{v:.4g}" for v in
                                    holm_correct([float(p) for p in sig.loc[m, "p_value"]])]
        print(sig.to_string(index=False))
        sig.to_csv(OUT_DIR / "stage3_significance.csv", index=False)

    # ---- Headroom recovery ----
    # What fraction of the oracle's advantage over the baseline does each
    # deployable strategy capture? R4 is a diagnostic ceiling, so this is the
    # natural way to express how much of the available gain is realised.
    headroom = pd.DataFrame()
    if {"R0_first", "R4_oracle"} <= set(df.strategy):
        piv = df.pivot_table(index="question_id", columns="strategy",
                             values=["gated_quality", "exact_match"])
        rows = []
        for metric in ["gated_quality", "exact_match"]:
            base = piv[(metric, "R0_first")]
            ceil = piv[(metric, "R4_oracle")]
            total = float((ceil - base).mean())
            for s in [x for x in [STRATEGY_R3, "R2_heuristic", "R1_random"]
                      if (metric, x) in piv.columns]:
                gain = float((piv[(metric, s)] - base).mean())
                rows.append({
                    "metric": metric, "strategy": s,
                    "gain_over_R0": round(gain, 4),
                    "total_headroom_R4_minus_R0": round(total, 4),
                    "headroom_recovered_pct": (round(gain / total * 100, 1)
                                               if total else None),
                })
        headroom = pd.DataFrame(rows)
        print_section_header("Headroom recovery (share of the R4 - R0 gap captured)")
        print(headroom.to_string(index=False))
        headroom.to_csv(OUT_DIR / "stage3_headroom_recovery.csv", index=False)

    # ---- Figure ----
    order = [s for s in list(STRATEGIES[:3]) + [STRATEGY_R3, "R4_oracle"]
             if s in summary.index]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    colours = {"R0_first": "#94a3b8", "R1_random": "#f59e0b",
               "R2_heuristic": "#a78bfa", "R3_llm": "#2563eb",
               "R4_oracle": "#16a34a"}
    for ax, metric, title in zip(axes, ["gated_quality", "exact_match"],
                                 ["Gated quality (PRIMARY, validated)",
                                  "Exact match (floor, known blind)"]):
        ax.bar(order, summary.loc[order, metric],
               color=[colours.get(s, "#64748b") for s in order], edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel("mean")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Stage 3 — candidate selection strategies "
                 "(green = oracle ceiling, grey = Stage 2 default)", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stage3_strategies.png", dpi=200, bbox_inches="tight")
    print(f"\n  [Saved] {FIG_DIR / 'stage3_strategies.png'}")

    write_report(summary, sig, headroom)
    print("\n  ✅ Stage 3 evaluation complete.")


def write_report(summary: pd.DataFrame, sig: pd.DataFrame,
                 headroom: pd.DataFrame = None) -> None:
    def md(f, index=True):
        if f is None or not len(f):
            return "_not computed_"
        try:
            return f.to_markdown(index=index)
        except ImportError:
            return "```\n" + f.to_string(index=index) + "\n```"

    verdict = "_primary comparison unavailable_"
    if len(sig):
        p = sig[(sig.family == "PRIMARY") & (sig.metric == "gated_quality")]
        if len(p):
            r = p.iloc[0]
            name = str(r["comparison"]).split(" vs ")[0]
            d, pv, n = float(r["delta"]), float(r["p_value"]), int(r["n_paired"])
            lo, hi = float(r["ci95_low"]), float(r["ci95_high"])

            rec = ""
            if headroom is not None and len(headroom):
                h = headroom[(headroom.metric == "gated_quality") &
                             (headroom.strategy == name)]
                if len(h):
                    pct = h.iloc[0]["headroom_recovered_pct"]
                    tot = h.iloc[0]["total_headroom_R4_minus_R0"]
                    rec = (f" It recovers **{pct}%** of the {tot:+.4f} "
                           f"oracle headroom (R4 - R0).")

            if pv < 0.05 and d > 0:
                verdict = (f"**{name} significantly improves distractor quality** "
                           f"over the Stage 2 default: {d:+.4f} (95% CI "
                           f"[{lo:+.4f}, {hi:+.4f}], p={pv:.4g}, n={n}).{rec} "
                           f"Re-ranking is a worthwhile pipeline component.")
            elif pv < 0.05 and d < 0:
                verdict = (f"**{name} is significantly WORSE than the baseline**: "
                           f"{d:+.4f} (95% CI [{lo:+.4f}, {hi:+.4f}], p={pv:.4g}, "
                           f"n={n}).{rec} Reordering by an uninformative signal "
                           f"discards the generator's own ordering, which "
                           f"Experiment 3.1 showed carries real quality signal "
                           f"(R0 > R1).")
            else:
                verdict = (f"**No significant improvement from {name}**: {d:+.4f} "
                           f"(95% CI [{lo:+.4f}, {hi:+.4f}], p={pv:.4g}, n={n}).{rec} "
                           f"This null comes from a validated instrument, not a "
                           f"blind one. A large remaining oracle gap would mean the "
                           f"headroom is real but this ranker does not capture it.")

    lines = [
        "# Stage 3 — Experiment 3.1: Candidate Re-ranking", "",
        "Applies hard validity filtering and four selection strategies to the "
        "existing Stage 2 generations. No candidates were regenerated; the only "
        "variable is the selection rule.", "",
        "| Strategy | Role |", "|---|---|",
        *[f"| {s} | {STRATEGY_DESCRIPTIONS[s]} |" for s in STRATEGIES], "",
        "## Results", "", md(summary), "",
        "## Paired comparisons", "",
        md(sig, index=False) if len(sig) else "_none_", "",
        "## Headroom recovery", "",
        "Share of the oracle's advantage over the baseline (R4 - R0) that each "
        "deployable strategy captures. R4 is a diagnostic ceiling, not a system.",
        "", md(headroom, index=False), "",
        "## Verdict", "", verdict, "",
        "## Method notes", "",
        "- **R3 ranks; it does not evaluate.** The pointwise LLM judge selects "
        "candidates, while evaluation uses the independent gated-quality "
        "composite. Using one instrument for both would be circular.",
        "- **Primary metric** is the validity-gated similarity composite, "
        "validated on known-quality conditions (AUC 0.898 good-vs-random, 0.995 "
        "good-vs-correct-answer) before use.",
        "- **Exact match is reported as a floor only.** It failed its positive "
        "control at the candidate level (AUC 0.507, 98.2% false-negative rate on "
        "known-good distractors) and no conclusion rests on it.",
        "- **R4 is a diagnostic ceiling, not a system.** It consults gold "
        "distractors and is not deployable; it bounds what any ranker could achieve.",
        "- **The heuristic uses no gold data and no fitted parameters** — its three "
        "features are equally weighted by fiat, so no test-set tuning is possible.",
        "- The Stage 1 retriever is not used to score candidates, avoiding circular "
        "evaluation.",
        "", "---", "*Auto-generated by scripts/15_evaluate_rerank.py.*",
    ]
    p = OUT_DIR / "stage3_report.md"
    p.write_text("\n".join(lines))
    print(f"  [Saved] {p}")


if __name__ == "__main__":
    main()
