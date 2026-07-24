#!/usr/bin/env python3
"""
Stage 06 — Experiment 2: checkpoint-trajectory evaluation.

Purpose: determine whether retrieval performance peaks earlier in training
than triplet validation accuracy, i.e. whether triplet accuracy is an
appropriate model-selection criterion.

Protocol (validation-first, no test leakage):
    1. Verify the training trajectory is identical to Experiment 1
       (loss + eval accuracy at common steps vs the committed reference log).
    2. Evaluate EVERY saved checkpoint on VALIDATION queries only,
       reusing the exact Experiment 1 retrieval pipeline.
    3. Select the best checkpoint by validation MRR@10
       (tie-break: validation HitRate@10).
    4. Evaluate ONLY that selected checkpoint on the TEST queries, once.

Outputs:
    outputs/results/checkpoint_metrics_val.csv
    outputs/results/final_test_results.csv
    outputs/results/checkpoint_trajectory_report.md
    outputs/figures/exp2_trajectory/*.png

Usage (on the training machine, after
       train_gpu.py --save-steps 50 --eval-steps 50 --keep-all-checkpoints):
    python scripts/06_evaluate_checkpoints.py
"""

import sys
import json
import argparse
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    MODELS_DIR, RESULTS_DIR, EMBEDDINGS_DIR, FIGURES_DIR, OUTPUT_DIR,
    BASELINE_MODEL_NAME,
)
from src.utils import set_seed, print_section_header, setup_matplotlib
from src.retrieval_experiment import run_retrieval_evaluation
from src.triplet_construction import load_triplets
from src.fine_tune import create_evaluator

K_VALUES = [10, 25]
# Maps our internal metric keys to the requested CSV column names.
COLUMN_MAP = {"hit_rate": "HitRate", "recall": "Recall", "mrr": "MRR", "ndcg": "nDCG"}
SELECTION_METRIC = "MRR@10"          # primary selection criterion (validation)
SELECTION_TIEBREAK = "HitRate@10"

TRAJ_RESULTS = RESULTS_DIR / "trajectory"
TRAJ_EMBEDDINGS = EMBEDDINGS_DIR / "trajectory"
FIG_DIR = FIGURES_DIR / "exp2_trajectory"
# Committed (tracked) Experiment 1 reference log for the identity check.
# Lives outside outputs/ because outputs/ is gitignored and would not reach
# a fresh Colab clone.
REFERENCE_LOG = PROJECT_ROOT / "reference" / "exp1_training_log.json"


def discover_checkpoints(model_dir: Path):
    """Return [(step, path)] for all saved checkpoints, sorted by step."""
    ckpt_dir = model_dir / "checkpoints"
    found = []
    for p in ckpt_dir.glob("checkpoint-*"):
        m = re.fullmatch(r"checkpoint-(\d+)", p.name)
        if m and (p / "model.safetensors").exists():
            found.append((int(m.group(1)), p))
    return sorted(found)


def trajectory_identity_check(model_dir: Path) -> dict:
    """
    Task 6: compare this run's training log against the Experiment 1
    reference at common steps. Returns a verdict dict; prints loudly.
    """
    result = {"status": "SKIPPED", "details": []}
    current_path = model_dir / "training_log.json"
    if not REFERENCE_LOG.exists() or not current_path.exists():
        missing = REFERENCE_LOG if not REFERENCE_LOG.exists() else current_path
        result["details"].append(f"log not found: {missing}")
        print(f"  ⚠ Identity check skipped — {missing} missing.")
        return result

    ref = json.load(open(REFERENCE_LOG))
    cur = json.load(open(current_path))

    def series(log, key):
        return {e["step"]: e[key] for e in log if key in e}

    diffs = []
    for key, label in [("loss", "train loss"), ("eval_val_cosine_accuracy", "val triplet accuracy")]:
        r, c = series(ref, key), series(cur, key)
        common = sorted(set(r) & set(c))
        if not common:
            diffs.append((label, None, "no common steps"))
            continue
        max_diff = max(abs(r[s] - c[s]) for s in common)
        diffs.append((label, max_diff, f"{len(common)} common steps"))

    max_all = max((d for _, d, _ in diffs if d is not None), default=None)
    if max_all is None:
        result["status"] = "INCOMPARABLE"
    elif max_all < 1e-4:
        result["status"] = "IDENTICAL"
    elif max_all < 5e-3:
        result["status"] = "NEGLIGIBLE_DIFFERENCE"
    else:
        result["status"] = "DIVERGED"

    print(f"  Trajectory identity vs Experiment 1: {result['status']}")
    for label, d, note in diffs:
        detail = f"{label}: max |diff| = {d:.6f} ({note})" if d is not None else f"{label}: {note}"
        result["details"].append(detail)
        print(f"    {detail}")
    if result["status"] == "DIVERGED":
        print("  ❌ WARNING: trajectories differ — Experiment 2 is NOT a faithful")
        print("     replay of Experiment 1. Results below describe THIS run only.")
    return result


def metrics_row(metrics: dict) -> dict:
    """Convert internal metric keys to the requested CSV columns."""
    row = {}
    for k in K_VALUES:
        for internal, public in COLUMN_MAP.items():
            row[f"{public}@{k}"] = metrics[f"{internal}@{k}"]
    return row


def triplet_accuracy_from_state(ckpt_path: Path, step: int):
    """Read this checkpoint's own val triplet accuracy from trainer_state.json."""
    state_file = ckpt_path / "trainer_state.json"
    if not state_file.exists():
        return None
    state = json.load(open(state_file))
    for e in state.get("log_history", []):
        if e.get("step") == step and "eval_val_cosine_accuracy" in e:
            return float(e["eval_val_cosine_accuracy"])
    return None


def baseline_triplet_accuracy() -> float:
    """Compute the pretrained model's accuracy on the val triplets (step-0 row)."""
    from sentence_transformers import SentenceTransformer
    val_triplets = load_triplets("val_triplets.jsonl")
    evaluator = create_evaluator(val_triplets)
    model = SentenceTransformer(BASELINE_MODEL_NAME)
    out = evaluator(model)
    if isinstance(out, dict):
        key = [k for k in out if "accuracy" in k][0]
        return float(out[key])
    return float(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints on retrieval")
    parser.add_argument("--model-dir", type=Path,
                        default=MODELS_DIR / "finetuned_pedagogical")
    parser.add_argument("--experiment-name", type=str, default="Experiment 2",
                        help="Name used in report title/prose")
    parser.add_argument("--skip-identity-check", action="store_true",
                        help="Skip the loss-identity check vs Experiment 1. Use for "
                             "Experiment 3+, where the objective changed by design so "
                             "the training loss is deliberately not comparable.")
    parser.add_argument("--compare-val-csv", type=Path, default=None,
                        help="Optional prior-experiment checkpoint_metrics_val.csv "
                             "(e.g. Experiment 2) to compare the retrieval trajectory against.")
    parser.add_argument("--fig-dir-name", type=str, default="exp2_trajectory",
                        help="Subdirectory under outputs/figures/ for the trajectory plots.")
    args = parser.parse_args()

    global FIG_DIR
    FIG_DIR = FIGURES_DIR / args.fig_dir_name

    set_seed()
    setup_matplotlib()
    TRAJ_RESULTS.mkdir(parents=True, exist_ok=True)
    TRAJ_EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print_section_header(f"{args.experiment_name.upper()}: Checkpoint Trajectory Evaluation")

    checkpoints = discover_checkpoints(args.model_dir)
    if not checkpoints:
        sys.exit(f"No checkpoints found under {args.model_dir}/checkpoints — "
                 f"run train_gpu.py --save-steps 50 --eval-steps 50 --keep-all-checkpoints first.")
    print(f"  Discovered {len(checkpoints)} checkpoints: "
          f"{[s for s, _ in checkpoints]}")

    # ---- Task 6: identity check (before spending any compute) ----
    if args.skip_identity_check:
        identity = {
            "status": "NOT APPLICABLE",
            "details": [
                "Training objective changed by design, so the training-loss "
                "trajectory is deliberately NOT comparable to Experiment 1.",
                "Controlled factors held fixed by construction: seed, dataset, "
                "split, source triplets, checkpoint cadence (every 50 steps).",
            ],
        }
        print("\n  Identity check skipped (objective changed by design).")
    else:
        print("\nTrajectory identity check...")
        identity = trajectory_identity_check(args.model_dir)

    # ---- Baseline reference (validation queries) ----
    print_section_header("Baseline reference on VALIDATION queries")
    base_val = run_retrieval_evaluation(
        model_name_or_path=BASELINE_MODEL_NAME, tag="baseline_val",
        query_split="val", results_dir=TRAJ_RESULTS,
        embeddings_dir=TRAJ_EMBEDDINGS, k_values=K_VALUES,
    )["metrics"]
    base_triplet_acc = baseline_triplet_accuracy()
    print(f"  Baseline val triplet accuracy: {base_triplet_acc:.4f}")

    # ---- Evaluate every checkpoint on VALIDATION queries ----
    rows = [{
        "checkpoint": "baseline (pretrained)",
        "training_step": 0,
        "triplet_validation_accuracy": base_triplet_acc,
        **metrics_row(base_val),
    }]
    for step, path in checkpoints:
        print_section_header(f"Checkpoint {step} — validation retrieval")
        res = run_retrieval_evaluation(
            model_name_or_path=str(path), tag=f"ckpt{step:04d}_val",
            query_split="val", results_dir=TRAJ_RESULTS,
            embeddings_dir=TRAJ_EMBEDDINGS, k_values=K_VALUES,
        )
        acc = triplet_accuracy_from_state(path, step)
        rows.append({
            "checkpoint": path.name,
            "training_step": step,
            "triplet_validation_accuracy": acc,
            **metrics_row(res["metrics"]),
        })

    df = pd.DataFrame(rows).sort_values("training_step").reset_index(drop=True)
    csv_path = RESULTS_DIR / "checkpoint_metrics_val.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  [Saved] {csv_path}")
    print(df.to_string(index=False))

    # ---- Task 3: validation-first selection ----
    ckpt_df = df[df["training_step"] > 0]
    best = ckpt_df.sort_values(
        [SELECTION_METRIC, SELECTION_TIEBREAK], ascending=False
    ).iloc[0]
    best_step = int(best["training_step"])
    best_path = dict(checkpoints)[best_step]
    print_section_header(
        f"Selected checkpoint (by val {SELECTION_METRIC}): step {best_step}")

    # ---- Single TEST evaluation of the selected checkpoint ----
    sel_test = run_retrieval_evaluation(
        model_name_or_path=str(best_path), tag="exp2_selected",
        query_split="test", results_dir=RESULTS_DIR,
        embeddings_dir=TRAJ_EMBEDDINGS, k_values=K_VALUES,
    )["metrics"]

    # Baseline TEST metrics: reuse saved artifact when available, else compute.
    base_test_path = RESULTS_DIR / "baseline_metrics.json"
    if base_test_path.exists():
        base_test = json.load(open(base_test_path))
        print("  Baseline test metrics: loaded from saved artifact.")
    else:
        print("  Baseline test metrics not found — computing (same pipeline)...")
        base_test = run_retrieval_evaluation(
            model_name_or_path=BASELINE_MODEL_NAME, tag="baseline",
            query_split="test", results_dir=RESULTS_DIR,
            embeddings_dir=EMBEDDINGS_DIR, k_values=K_VALUES,
        )["metrics"]

    base_row, sel_row = metrics_row(base_test), metrics_row(sel_test)
    final_rows = []
    for col in base_row:
        b, s = base_row[col], sel_row[col]
        final_rows.append({
            "metric": col, "baseline": round(b, 4),
            f"selected (step {best_step})": round(s, 4),
            "delta_abs": round(s - b, 4),
            "delta_pct": round((s - b) / b * 100, 2) if b else None,
        })
    final_df = pd.DataFrame(final_rows)
    final_csv = RESULTS_DIR / "final_test_results.csv"
    final_df.to_csv(final_csv, index=False)
    print(f"\n  [Saved] {final_csv}")
    print(final_df.to_string(index=False))

    # ---- Optional prior-experiment trajectory for comparison overlay ----
    compare_df = None
    if args.compare_val_csv and args.compare_val_csv.exists():
        compare_df = pd.read_csv(args.compare_val_csv)
        print(f"\n  Loaded comparison trajectory: {args.compare_val_csv}")

    # ---- Task 5: figures ----
    steps = df["training_step"].values
    base_mask = steps == 0
    plots = [
        ("triplet_validation_accuracy", "Triplet validation accuracy", "triplet_accuracy"),
        ("HitRate@10", "Validation HitRate@10", "val_hitrate10"),
        ("Recall@10", "Validation Recall@10", "val_recall10"),
        ("MRR@10", "Validation MRR@10", "val_mrr10"),
        ("nDCG@10", "Validation nDCG@10", "val_ndcg10"),
    ]
    for col, title, fname in plots:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        vals = df[col].astype(float).values
        # Prior-experiment overlay (same column, if present)
        if compare_df is not None and col in compare_df.columns:
            c = compare_df[compare_df["training_step"] > 0]
            ax.plot(c["training_step"], c[col].astype(float), "s--", color="#f59e0b",
                    alpha=0.7, label="prior experiment (comparison)")
        ax.plot(steps[~base_mask], vals[~base_mask], "o-", color="#2563eb",
                label=f"{args.experiment_name} trajectory")
        ax.axhline(vals[base_mask][0], color="#0f172a", ls="--", lw=1.2,
                   label=f"baseline (pretrained) = {vals[base_mask][0]:.4f}")
        ax.axvline(best_step, color="#16a34a", ls=":", lw=1.2,
                   label=f"selected checkpoint (step {best_step})")
        ax.set_xlabel("training step")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs training step — {args.experiment_name}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{fname}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"\n  [Saved] 5 figures to {FIG_DIR}")

    # ---- Task 7: report ----
    write_report(df, identity, best_step, base_test, sel_test, final_df,
                 experiment_name=args.experiment_name, compare_df=compare_df)
    print(f"\n  ✅ {args.experiment_name} evaluation complete.")


def _df_to_md(frame: pd.DataFrame) -> str:
    """DataFrame to markdown table; falls back to a code block without tabulate."""
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```\n" + frame.to_string(index=False) + "\n```"


def write_report(df, identity, best_step, base_test, sel_test, final_df,
                 experiment_name: str = "Experiment 2", compare_df=None) -> None:
    """Auto-generate the trajectory report answering the 7 questions."""
    ck = df[df["training_step"] > 0].reset_index(drop=True)
    base = df[df["training_step"] == 0].iloc[0]

    early = ck[ck["training_step"] <= 200]
    q1_improves = bool((early["MRR@10"] > base["MRR@10"]).any()) if len(early) else False

    retr_peak_step = int(ck.loc[ck["MRR@10"].idxmax(), "training_step"])
    acc_peak_step = int(ck.loc[ck["triplet_validation_accuracy"].idxmax(), "training_step"])
    retr_peak = ck["MRR@10"].max()
    last = ck.iloc[-1]
    degrades_after_peak = bool(last["MRR@10"] < retr_peak - 1e-6)

    acc_best_row = ck.loc[ck["triplet_validation_accuracy"].idxmax()]
    retr_best_row = ck.loc[ck["MRR@10"].idxmax()]
    selection_cost = acc_best_row["MRR@10"] - retr_best_row["MRR@10"]

    beats_baseline_val = retr_peak > base["MRR@10"]
    sel_beats_baseline_test = sel_test["mrr@10"] > base_test["mrr@10"]

    if not beats_baseline_val:
        next_exp = (
            "No checkpoint beats the pretrained baseline on validation retrieval, "
            "so this single-factor change was not sufficient. **The next experiment "
            "should change one different factor**, targeting the most-implicated "
            "remaining causes: (a) full-parameter fine-tuning destroying pretrained "
            "structure — test by freezing the encoder and training only a projection "
            "head (or LoRA); (b) supervision sparsity in the long tail. Keep data, "
            "split, model, and schedule fixed."
        )
    elif sel_beats_baseline_test:
        next_exp = (
            "Validation-selected fine-tuning beats the baseline on the held-out "
            "test set. **The next experiment should hold everything fixed and change "
            "only negative hardness** (mined hard negatives) to push the "
            "top-of-ranking margin further."
        )
    else:
        next_exp = (
            "A checkpoint beats baseline on validation but not on test — "
            "consistent with selection overfitting to the small validation set. "
            "**The next experiment should quantify selection variance** (e.g., "
            "repeated selection over bootstrap resamples of validation queries) "
            "before any architectural change."
        )

    # Optional comparison against a prior experiment's trajectory
    compare_lines = []
    if compare_df is not None:
        pc = compare_df[compare_df["training_step"] > 0]
        prior_best = pc["MRR@10"].max()
        prior_best_step = int(pc.loc[pc["MRR@10"].idxmax(), "training_step"])
        verdict = ("higher" if retr_peak > prior_best
                   else "lower" if retr_peak < prior_best else "equal")
        compare_lines = [
            "## Trajectory comparison vs prior experiment",
            f"Best val MRR@10 — this experiment: **{retr_peak:.4f}** (step {retr_peak_step}); "
            f"prior experiment: **{prior_best:.4f}** (step {prior_best_step}). "
            f"This experiment's best checkpoint is **{verdict}** than the prior experiment's, "
            f"and baseline is {base['MRR@10']:.4f}.",
            "",
        ]

    lines = [
        f"# {experiment_name} — Checkpoint Trajectory Report",
        "",
        f"Selection criterion: validation {SELECTION_METRIC} "
        f"(tie-break {SELECTION_TIEBREAK}). Test split touched exactly once.",
        "",
        "## Trajectory identity / controlled-factor check",
        f"**Status: {identity['status']}**",
        *[f"- {d}" for d in identity["details"]],
        "",
        *compare_lines,
        "## Q1 — Does retrieval improve during early training?",
        f"{'**Yes**' if q1_improves else '**No**'}: within the first 200 steps, "
        f"best val MRR@10 = {early['MRR@10'].max():.4f} vs baseline "
        f"{base['MRR@10']:.4f}." if len(early) else "No checkpoints ≤ 200 steps.",
        "",
        "## Q2 — Does retrieval peak before triplet validation accuracy?",
        f"Retrieval (val MRR@10) peaks at **step {retr_peak_step}**; triplet "
        f"validation accuracy peaks at **step {acc_peak_step}**. "
        f"{'**Yes — retrieval peaks earlier.**' if retr_peak_step < acc_peak_step else '**No.**'}",
        "",
        "## Q3 — Does retrieval later degrade?",
        f"{'**Yes**' if degrades_after_peak else '**No**'}: final checkpoint "
        f"(step {int(last['training_step'])}) val MRR@10 = {last['MRR@10']:.4f} "
        f"vs peak {retr_peak:.4f}.",
        "",
        "## Q4 — Is triplet validation accuracy an appropriate selection criterion?",
        f"Selecting by triplet accuracy picks step {acc_peak_step} "
        f"(val MRR@10 {acc_best_row['MRR@10']:.4f}); selecting by retrieval picks "
        f"step {retr_peak_step} (val MRR@10 {retr_best_row['MRR@10']:.4f}). "
        f"Cost of the triplet-accuracy criterion: {selection_cost:+.4f} val MRR@10. "
        + ("**The criteria agree** — triplet accuracy was adequate here."
           if retr_peak_step == acc_peak_step else
           "**The criteria disagree — triplet accuracy is a misaligned proxy.**"),
        "",
        "## Q5 — Best checkpoint on validation retrieval",
        f"**checkpoint-{best_step}** (val MRR@10 = {retr_best_row['MRR@10']:.4f}, "
        f"val HitRate@10 = {retr_best_row['HitRate@10']:.4f}).",
        "",
        "## Q6 — Final TEST metrics of the selected checkpoint",
        "",
        _df_to_md(final_df),
        "",
        f"Selected checkpoint {'**beats**' if sel_beats_baseline_test else '**does not beat**'} "
        f"the baseline on test MRR@10 "
        f"({sel_test['mrr@10']:.4f} vs {base_test['mrr@10']:.4f}).",
        "",
        "## Q7 — What should the next experiment investigate?",
        next_exp,
        "",
        "---",
        "*Auto-generated by scripts/06_evaluate_checkpoints.py; every number "
        "derives from checkpoint_metrics_val.csv and final_test_results.csv.*",
    ]
    report_path = RESULTS_DIR / "checkpoint_trajectory_report.md"
    report_path.write_text("\n".join(lines))
    print(f"  [Saved] {report_path}")


if __name__ == "__main__":
    main()
