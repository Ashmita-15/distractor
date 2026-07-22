#!/usr/bin/env python3
"""
Main pipeline for Misconception-Aware Distractor Generation.

Executes all 9 phases sequentially:
  Phase 1: Dataset Understanding
  Phase 2: Research-Oriented EDA
  Phase 3: Data Preparation
  Phase 4: Baseline Semantic Retrieval
  Phase 5: Baseline Evaluation
  Phase 6: Triplet Construction
  Phase 7: Fine-tune SentenceTransformer
  Phase 8: Evaluate Fine-tuned Model
  Phase 9: Comparison

Usage:
    python scripts/run_pipeline.py [--phases 1,2,3,4,5,6,7,8,9]

Each phase validates its outputs before proceeding to the next.
"""

import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FIGURES_DIR, EMBEDDINGS_DIR, MODELS_DIR, RESULTS_DIR,
    TOP_K_VALUES, MAX_K, RANDOM_SEED,
)
from src.utils import (
    set_seed, setup_matplotlib, save_figure, save_results,
    print_section_header, get_device,
)
from src.data_loader import (
    load_raw_data, validate_data_integrity, melt_to_qdp,
    create_text_representation, get_misconception_stats,
)
from src.eda import run_full_eda
from src.data_preparation import create_splits
from src.evaluation import (
    compute_retrieval_metrics, compute_stratified_metrics,
    format_metrics_table, compare_models,
)


def phase_1():
    """Phase 1: Dataset Understanding."""
    print_section_header("PHASE 1: Dataset Understanding")

    print("Loading raw data...")
    train_df, test_df, misconception_df = load_raw_data()

    print("\nValidating data integrity...")
    integrity = validate_data_integrity(train_df, misconception_df)

    print("\nMelting to Question-Distractor Pairs (QDPs)...")
    qdp_df = melt_to_qdp(train_df, misconception_df, keep_unlabeled=False)

    print("\nCreating text representations...")
    qdp_df = create_text_representation(qdp_df)

    print("\nMisconception statistics...")
    misc_stats = get_misconception_stats(qdp_df)
    print(f"  Top 10 misconceptions by frequency:")
    print(misc_stats.head(10)[["MisconceptionId", "count", "n_unique_constructs", "misconception_name"]].to_string(index=False))

    return train_df, test_df, misconception_df, qdp_df


def phase_2(train_df, qdp_df, misconception_df):
    """Phase 2: Research-Oriented EDA."""
    print_section_header("PHASE 2: Research-Oriented EDA")

    eda_stats = run_full_eda(train_df, qdp_df, misconception_df)

    # Save EDA statistics
    save_results(eda_stats, "eda_statistics.json", RESULTS_DIR)

    print("\n  ✅ EDA complete. Figures saved to:", FIGURES_DIR)
    return eda_stats


def phase_3(qdp_df):
    """Phase 3: Data Preparation (Split)."""
    print_section_header("PHASE 3: Data Preparation")

    print("Creating question-level stratified splits...")
    train_qdp, val_qdp, test_qdp = create_splits(qdp_df)

    # Save splits
    train_qdp.to_csv(RESULTS_DIR / "train_qdp.csv", index=False)
    val_qdp.to_csv(RESULTS_DIR / "val_qdp.csv", index=False)
    test_qdp.to_csv(RESULTS_DIR / "test_qdp.csv", index=False)
    print(f"\n  ✅ Splits saved to {RESULTS_DIR}")

    return train_qdp, val_qdp, test_qdp


def phase_4(train_qdp, val_qdp, test_qdp):
    """Phase 4: Baseline Semantic Retrieval."""
    print_section_header("PHASE 4: Baseline Semantic Retrieval")

    from src.baseline_retrieval import encode_texts, build_faiss_index, save_embeddings

    # Use train+val as corpus, test as queries
    # For fair evaluation, the retrieval corpus is train+val
    corpus_qdp = pd.concat([train_qdp, val_qdp], ignore_index=True)
    query_qdp = test_qdp

    print(f"  Corpus size: {len(corpus_qdp):,} QDPs")
    print(f"  Query size:  {len(query_qdp):,} QDPs")

    # Encode corpus
    print("\nEncoding corpus...")
    corpus_embeddings = encode_texts(corpus_qdp["text"].tolist())
    save_embeddings(corpus_embeddings, "baseline_corpus_embeddings.npy")

    # Encode queries
    print("\nEncoding queries...")
    query_embeddings = encode_texts(query_qdp["text"].tolist())
    save_embeddings(query_embeddings, "baseline_query_embeddings.npy")

    # Build FAISS index
    print("\nBuilding FAISS index...")
    index = build_faiss_index(corpus_embeddings)

    # Retrieve
    from src.baseline_retrieval import retrieve_top_k
    print("\nRetrieving top-K neighbors...")
    scores, indices = retrieve_top_k(
        query_embeddings=query_embeddings,
        index=index,
        k=MAX_K,
        query_question_ids=query_qdp["QuestionId"].values,
        corpus_question_ids=corpus_qdp["QuestionId"].values,
    )

    print(f"  Retrieved shape: {indices.shape}")

    return corpus_qdp, query_qdp, corpus_embeddings, query_embeddings, scores, indices


def phase_5(corpus_qdp, query_qdp, indices):
    """Phase 5: Baseline Evaluation."""
    print_section_header("PHASE 5: Baseline Evaluation")

    query_misc_ids = query_qdp["MisconceptionId"].values
    corpus_misc_ids = corpus_qdp["MisconceptionId"].values

    # Map retrieved indices to misconception IDs
    retrieved_misc_ids = corpus_misc_ids[indices]

    # Overall metrics
    print("Computing overall metrics...")
    baseline_metrics = compute_retrieval_metrics(
        query_misconception_ids=query_misc_ids,
        retrieved_misconception_ids=retrieved_misc_ids,
        corpus_misconception_ids=corpus_misc_ids,
        k_values=TOP_K_VALUES,
    )

    metrics_table = format_metrics_table(baseline_metrics, TOP_K_VALUES, "Baseline")
    print("\n  === Baseline Retrieval Metrics ===")
    print(metrics_table.to_string(index=False))

    # Stratified metrics
    print("\n  Computing stratified metrics by misconception frequency...")
    stratified = compute_stratified_metrics(
        query_misconception_ids=query_misc_ids,
        retrieved_misconception_ids=retrieved_misc_ids,
        corpus_misconception_ids=corpus_misc_ids,
        k_values=TOP_K_VALUES,
    )

    print("\n  === Stratified Results ===")
    for bucket, metrics in stratified.items():
        n = metrics.get("n_queries", 0)
        hr10 = metrics.get("hit_rate@10", 0)
        mrr10 = metrics.get("mrr@10", 0)
        print(f"    {bucket:20s}  n={n:4d}  HR@10={hr10:.4f}  MRR@10={mrr10:.4f}")

    # Save results
    save_results(baseline_metrics, "baseline_metrics.json", RESULTS_DIR)
    save_results(stratified, "baseline_stratified_metrics.json", RESULTS_DIR)

    print("\n  ✅ Baseline evaluation complete.")
    return baseline_metrics, stratified


def phase_6(train_qdp, val_qdp):
    """Phase 6: Triplet Construction."""
    print_section_header("PHASE 6: Triplet Construction")

    from src.triplet_construction import construct_triplets, validate_triplets

    # Construct training triplets
    print("Constructing training triplets (in-subject negatives)...")
    train_triplets = construct_triplets(
        train_qdp, strategy="in_subject", negatives_per_anchor=3, seed=RANDOM_SEED,
    )

    # Construct validation triplets
    print("\nConstructing validation triplets (in-subject negatives)...")
    val_triplets = construct_triplets(
        val_qdp, strategy="in_subject", negatives_per_anchor=2, seed=RANDOM_SEED + 1,
    )

    # Validate
    print("\n  === Training Triplet Samples ===")
    validate_triplets(train_triplets, n_samples=3)

    print(f"\n  ✅ Triplet construction complete.")
    print(f"      Training triplets: {len(train_triplets):,}")
    print(f"      Validation triplets: {len(val_triplets):,}")

    return train_triplets, val_triplets


def phase_7(train_triplets, val_triplets):
    """Phase 7: Fine-tune SentenceTransformer."""
    print_section_header("PHASE 7: Fine-tune SentenceTransformer")

    from src.fine_tune import finetune_model

    model, model_path = finetune_model(
        train_triplets=train_triplets,
        val_triplets=val_triplets,
        output_dir=MODELS_DIR / "finetuned_pedagogical",
    )

    print(f"\n  ✅ Fine-tuning complete. Model saved to: {model_path}")
    return model, model_path


def phase_8(model_path, corpus_qdp, query_qdp):
    """Phase 8: Evaluate Fine-tuned Model."""
    print_section_header("PHASE 8: Evaluate Fine-tuned Model")

    from src.baseline_retrieval import encode_texts, build_faiss_index, retrieve_top_k, save_embeddings

    # Encode with fine-tuned model
    print("Encoding corpus with fine-tuned model...")
    ft_corpus_embeddings = encode_texts(
        corpus_qdp["text"].tolist(),
        model_name_or_path=str(model_path),
    )
    save_embeddings(ft_corpus_embeddings, "finetuned_corpus_embeddings.npy")

    print("\nEncoding queries with fine-tuned model...")
    ft_query_embeddings = encode_texts(
        query_qdp["text"].tolist(),
        model_name_or_path=str(model_path),
    )
    save_embeddings(ft_query_embeddings, "finetuned_query_embeddings.npy")

    # Build index and retrieve
    print("\nBuilding FAISS index...")
    ft_index = build_faiss_index(ft_corpus_embeddings)

    print("\nRetrieving top-K neighbors...")
    ft_scores, ft_indices = retrieve_top_k(
        query_embeddings=ft_query_embeddings,
        index=ft_index,
        k=MAX_K,
        query_question_ids=query_qdp["QuestionId"].values,
        corpus_question_ids=corpus_qdp["QuestionId"].values,
    )

    # Evaluate
    query_misc_ids = query_qdp["MisconceptionId"].values
    corpus_misc_ids = corpus_qdp["MisconceptionId"].values
    retrieved_misc_ids = corpus_misc_ids[ft_indices]

    print("\nComputing overall metrics...")
    ft_metrics = compute_retrieval_metrics(
        query_misconception_ids=query_misc_ids,
        retrieved_misconception_ids=retrieved_misc_ids,
        corpus_misconception_ids=corpus_misc_ids,
        k_values=TOP_K_VALUES,
    )

    metrics_table = format_metrics_table(ft_metrics, TOP_K_VALUES, "Fine-tuned")
    print("\n  === Fine-tuned Retrieval Metrics ===")
    print(metrics_table.to_string(index=False))

    # Stratified
    print("\n  Computing stratified metrics...")
    ft_stratified = compute_stratified_metrics(
        query_misconception_ids=query_misc_ids,
        retrieved_misconception_ids=retrieved_misc_ids,
        corpus_misconception_ids=corpus_misc_ids,
        k_values=TOP_K_VALUES,
    )

    print("\n  === Stratified Results ===")
    for bucket, metrics in ft_stratified.items():
        n = metrics.get("n_queries", 0)
        hr10 = metrics.get("hit_rate@10", 0)
        mrr10 = metrics.get("mrr@10", 0)
        print(f"    {bucket:20s}  n={n:4d}  HR@10={hr10:.4f}  MRR@10={mrr10:.4f}")

    save_results(ft_metrics, "finetuned_metrics.json", RESULTS_DIR)
    save_results(ft_stratified, "finetuned_stratified_metrics.json", RESULTS_DIR)

    return ft_metrics, ft_stratified, ft_corpus_embeddings, ft_query_embeddings


def phase_9(
    baseline_metrics, ft_metrics,
    baseline_stratified, ft_stratified,
    corpus_qdp, query_qdp,
    baseline_corpus_emb, ft_corpus_emb,
    baseline_query_emb, ft_query_emb,
):
    """Phase 9: Comparison of Baseline vs Fine-tuned."""
    print_section_header("PHASE 9: Comparison — Baseline vs Fine-tuned")

    setup_matplotlib()

    # 1. Side-by-side metric comparison table
    print("  === Side-by-Side Comparison ===\n")
    comparison_df = compare_models(baseline_metrics, ft_metrics, TOP_K_VALUES)
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)

    # 2. Bar chart comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metric_names = ["hit_rate", "recall", "mrr", "ndcg"]
    titles = ["Hit Rate@K", "Recall@K", "MRR@K", "nDCG@K"]
    colors_base = "#94a3b8"
    colors_ft = "#2563eb"

    for ax, metric, title in zip(axes.flat, metric_names, titles):
        base_vals = [baseline_metrics.get(f"{metric}@{k}", 0) for k in TOP_K_VALUES]
        ft_vals = [ft_metrics.get(f"{metric}@{k}", 0) for k in TOP_K_VALUES]

        x = np.arange(len(TOP_K_VALUES))
        width = 0.35

        bars1 = ax.bar(x - width/2, base_vals, width, label="Baseline", color=colors_base, edgecolor="white")
        bars2 = ax.bar(x + width/2, ft_vals, width, label="Fine-tuned", color=colors_ft, edgecolor="white")

        ax.set_xlabel("K")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in TOP_K_VALUES])
        ax.legend()
        ax.set_ylim(0, max(max(base_vals), max(ft_vals)) * 1.2 + 0.05)

    fig.suptitle("Baseline vs Fine-tuned Retrieval: All Metrics", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "comparison_all_metrics.png", FIGURES_DIR)

    # 3. Stratified comparison chart (Hit Rate@10)
    fig, ax = plt.subplots(figsize=(10, 6))
    buckets = list(baseline_stratified.keys())
    base_hr10 = [baseline_stratified[b].get("hit_rate@10", 0) for b in buckets]
    ft_hr10 = [ft_stratified[b].get("hit_rate@10", 0) for b in buckets]
    n_queries = [baseline_stratified[b].get("n_queries", 0) for b in buckets]

    x = np.arange(len(buckets))
    width = 0.35

    ax.bar(x - width/2, base_hr10, width, label="Baseline", color=colors_base, edgecolor="white")
    ax.bar(x + width/2, ft_hr10, width, label="Fine-tuned", color=colors_ft, edgecolor="white")

    ax.set_xlabel("Misconception Frequency Bucket")
    ax.set_ylabel("Hit Rate@10")
    ax.set_title("Stratified Hit Rate@10: Does Fine-tuning Help Rare Misconceptions?")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(buckets, n_queries)], fontsize=9)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "stratified_comparison_hr10.png", FIGURES_DIR)

    # 4. t-SNE visualization (subsample for speed)
    print("\n  Generating t-SNE visualizations...")
    from sklearn.manifold import TSNE

    # Use corpus embeddings, subsample if large
    max_tsne = 1500
    n_corpus = len(corpus_qdp)
    if n_corpus > max_tsne:
        rng = np.random.RandomState(RANDOM_SEED)
        tsne_indices = rng.choice(n_corpus, max_tsne, replace=False)
    else:
        tsne_indices = np.arange(n_corpus)

    tsne_misc_ids = corpus_qdp.iloc[tsne_indices]["MisconceptionId"].values

    # Get top-10 most frequent misconceptions for coloring
    misc_counts = Counter(tsne_misc_ids)
    top_10_miscs = [m for m, _ in misc_counts.most_common(10)]
    color_map = {m: i for i, m in enumerate(top_10_miscs)}

    colors = []
    for m in tsne_misc_ids:
        if m in color_map:
            colors.append(color_map[m])
        else:
            colors.append(-1)  # "other"
    colors = np.array(colors)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    cmap = matplotlib.colormaps["tab10"]

    for ax, emb, title in zip(axes, [baseline_corpus_emb, ft_corpus_emb], ["Baseline", "Fine-tuned"]):
        sub_emb = emb[tsne_indices]
        tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30, max_iter=1000)
        proj = tsne.fit_transform(sub_emb)

        # Plot "other" in gray
        other_mask = colors == -1
        ax.scatter(proj[other_mask, 0], proj[other_mask, 1],
                   c="#d1d5db", s=8, alpha=0.3, label="Other")

        # Plot top misconceptions with colors
        for m_id in top_10_miscs:
            mask = colors == color_map[m_id]
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       c=[cmap(color_map[m_id])], s=20, alpha=0.7,
                       label=f"Misc {m_id}")

        ax.set_title(f"{title} Embeddings")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, markerscale=2)
    fig.suptitle("t-SNE Visualization: Misconception Clusters (Top 10 Misconceptions)", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "tsne_comparison.png", FIGURES_DIR)

    # 5. Statistical significance (Wilcoxon signed-rank test)
    print("\n  Running statistical significance test...")
    from scipy import stats as scipy_stats

    # Per-query Hit Rate@10
    query_misc_ids = query_qdp["MisconceptionId"].values

    # Reload retrieval results to compute per-query metrics
    # We'll compute per-query hit rates from saved results
    base_per_query = []
    ft_per_query = []

    # Reload indices from phase 4 and phase 8
    base_corpus_misc = corpus_qdp["MisconceptionId"].values

    from src.baseline_retrieval import build_faiss_index, retrieve_top_k, load_embeddings

    # Baseline per-query hit@10
    base_index = build_faiss_index(baseline_corpus_emb)
    base_scores, base_idx = retrieve_top_k(
        baseline_query_emb, base_index, k=10,
        query_question_ids=query_qdp["QuestionId"].values,
        corpus_question_ids=corpus_qdp["QuestionId"].values,
    )
    for i in range(len(query_qdp)):
        hits = (base_corpus_misc[base_idx[i]] == query_misc_ids[i]).any()
        base_per_query.append(float(hits))

    # Fine-tuned per-query hit@10
    ft_index = build_faiss_index(ft_corpus_emb)
    ft_scores, ft_idx = retrieve_top_k(
        ft_query_emb, ft_index, k=10,
        query_question_ids=query_qdp["QuestionId"].values,
        corpus_question_ids=corpus_qdp["QuestionId"].values,
    )
    for i in range(len(query_qdp)):
        hits = (base_corpus_misc[ft_idx[i]] == query_misc_ids[i]).any()
        ft_per_query.append(float(hits))

    base_per_query = np.array(base_per_query)
    ft_per_query = np.array(ft_per_query)

    # McNemar's exact test — the appropriate test for paired binary outcomes.
    # b = queries the baseline hits but the fine-tuned model misses;
    # c = queries the fine-tuned model hits but the baseline misses.
    # Under H0 (no difference) discordant pairs are Binomial(b+c, 0.5).
    b = int(((base_per_query == 1) & (ft_per_query == 0)).sum())
    c = int(((base_per_query == 0) & (ft_per_query == 1)).sum())
    if b + c > 0:
        mcnemar_p = scipy_stats.binomtest(c, b + c, 0.5, alternative="greater").pvalue
    else:
        mcnemar_p = 1.0
    print(f"    McNemar exact test on per-query Hit@10 (one-sided, fine-tuned > baseline):")
    print(f"    Discordant pairs: baseline-only hits b={b}, fine-tuned-only hits c={c}")
    print(f"    p-value:   {mcnemar_p:.6f}")
    print(f"    Significant at α=0.05: {'Yes ✅' if mcnemar_p < 0.05 else 'No ❌'}")

    # Wilcoxon signed-rank test (reported for reference)
    if not np.array_equal(base_per_query, ft_per_query):
        stat, p_value = scipy_stats.wilcoxon(ft_per_query, base_per_query, alternative="greater")
        print(f"\n    Wilcoxon signed-rank test (one-sided, fine-tuned > baseline):")
        print(f"    Statistic: {stat:.4f}")
        print(f"    p-value:   {p_value:.6f}")
    else:
        p_value = 1.0
        print("    Per-query results are identical — Wilcoxon not applicable.")

    # Save final comparison
    final_results = {
        "baseline_metrics": baseline_metrics,
        "finetuned_metrics": ft_metrics,
        "baseline_stratified": baseline_stratified,
        "finetuned_stratified": ft_stratified,
        "mcnemar_discordant_baseline_only": b,
        "mcnemar_discordant_finetuned_only": c,
        "mcnemar_p_value": float(mcnemar_p),
        "wilcoxon_p_value": float(p_value),
    }
    save_results(final_results, "final_comparison.json", RESULTS_DIR)

    print(f"\n  ✅ Phase 9 complete. All results saved to {RESULTS_DIR}")
    return final_results


def main():
    """Run the full pipeline."""
    parser = argparse.ArgumentParser(description="Misconception-Aware Retrieval Pipeline")
    parser.add_argument(
        "--phases", type=str, default="1,2,3,4,5,6,7,8,9",
        help="Comma-separated list of phases to run (default: all)"
    )
    args = parser.parse_args()
    phases_to_run = [int(p) for p in args.phases.split(",")]

    set_seed()

    # Phase 1
    if 1 in phases_to_run:
        train_df, test_df, misconception_df, qdp_df = phase_1()

    # Phase 2
    if 2 in phases_to_run:
        eda_stats = phase_2(train_df, qdp_df, misconception_df)

    # Phase 3
    if 3 in phases_to_run:
        train_qdp, val_qdp, test_qdp = phase_3(qdp_df)

    # Phase 4
    if 4 in phases_to_run:
        corpus_qdp, query_qdp, corpus_emb, query_emb, scores, indices = phase_4(
            train_qdp, val_qdp, test_qdp
        )

    # Phase 5
    if 5 in phases_to_run:
        baseline_metrics, baseline_stratified = phase_5(corpus_qdp, query_qdp, indices)

    # Phase 6
    if 6 in phases_to_run:
        train_triplets, val_triplets = phase_6(train_qdp, val_qdp)

    # Phase 7
    if 7 in phases_to_run:
        model, model_path = phase_7(train_triplets, val_triplets)

    # Phase 8
    if 8 in phases_to_run:
        ft_metrics, ft_stratified, ft_corpus_emb, ft_query_emb = phase_8(
            model_path, corpus_qdp, query_qdp
        )

    # Phase 9
    if 9 in phases_to_run:
        final_results = phase_9(
            baseline_metrics, ft_metrics,
            baseline_stratified, ft_stratified,
            corpus_qdp, query_qdp,
            corpus_emb, ft_corpus_emb,
            query_emb, ft_query_emb,
        )

    print_section_header("PIPELINE COMPLETE")
    print("  All phases executed successfully.")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Models:  {MODELS_DIR}")


if __name__ == "__main__":
    main()
