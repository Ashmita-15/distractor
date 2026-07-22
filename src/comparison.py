"""
Baseline vs fine-tuned comparison: tables, statistical tests, figures.

All inputs are artifacts saved by src.retrieval_experiment — nothing is
re-encoded here, so the comparison is cheap and exactly reflects the saved
evaluation runs.

Statistical testing:
    Per-query Hit@K is a paired binary outcome, so the primary test is
    McNemar's exact test on discordant pairs (baseline-only hits b vs
    fine-tuned-only hits c; under H0 discordants ~ Binomial(b+c, 0.5)).
    Wilcoxon signed-rank is reported for reference only.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from typing import Dict, Tuple

from scipy import stats as scipy_stats

from src.config import (
    RESULTS_DIR, EMBEDDINGS_DIR, FIGURES_DIR,
    TOP_K_VALUES, RANDOM_SEED,
)
from src.utils import save_results, save_figure, setup_matplotlib
from src.evaluation import compare_models


def load_artifacts(tag: str) -> Dict:
    """Load saved metrics and retrieval indices for a model tag."""
    with open(RESULTS_DIR / f"{tag}_metrics.json") as f:
        metrics = json.load(f)
    with open(RESULTS_DIR / f"{tag}_stratified_metrics.json") as f:
        stratified = json.load(f)
    indices = np.load(RESULTS_DIR / f"{tag}_indices.npy")
    return {"metrics": metrics, "stratified": stratified, "indices": indices}


def per_query_hits(
    indices: np.ndarray,
    query_misc: np.ndarray,
    corpus_misc: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Binary per-query Hit@K vector from saved retrieval indices."""
    retrieved = corpus_misc[indices[:, :k]]
    return (retrieved == query_misc[:, None]).any(axis=1).astype(float)


def mcnemar_test(base_hits: np.ndarray, ft_hits: np.ndarray) -> Tuple[int, int, float]:
    """
    One-sided McNemar exact test (fine-tuned > baseline).

    Returns:
        (b, c, p_value) where b = baseline-only hits, c = fine-tuned-only hits.
    """
    b = int(((base_hits == 1) & (ft_hits == 0)).sum())
    c = int(((base_hits == 0) & (ft_hits == 1)).sum())
    if b + c == 0:
        return b, c, 1.0
    p = scipy_stats.binomtest(c, b + c, 0.5, alternative="greater").pvalue
    return b, c, float(p)


def run_comparison(
    baseline_tag: str = "baseline",
    finetuned_tag: str = "finetuned",
    hit_k: int = 10,
    tsne_max_points: int = 1500,
) -> Dict:
    """
    Full comparison between two saved evaluation runs.

    Produces:
        results/comparison_table.csv
        results/final_comparison.json
        figures/comparison_all_metrics.png
        figures/stratified_comparison_hr10.png
        figures/tsne_comparison.png

    Returns:
        Dict of all comparison results.
    """
    from src.retrieval_experiment import load_corpus_and_queries

    setup_matplotlib()

    base = load_artifacts(baseline_tag)
    ft = load_artifacts(finetuned_tag)
    corpus_df, query_df, _ = load_corpus_and_queries("test")

    # --- 1. Side-by-side table ---
    print("  === Side-by-Side Comparison ===\n")
    comparison_df = compare_models(base["metrics"], ft["metrics"], TOP_K_VALUES)
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)

    # --- 2. Bar chart of all metrics ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metric_names = ["hit_rate", "recall", "mrr", "ndcg"]
    titles = ["Hit Rate@K", "Recall@K", "MRR@K", "nDCG@K"]
    for ax, metric, title in zip(axes.flat, metric_names, titles):
        base_vals = [base["metrics"].get(f"{metric}@{k}", 0) for k in TOP_K_VALUES]
        ft_vals = [ft["metrics"].get(f"{metric}@{k}", 0) for k in TOP_K_VALUES]
        x = np.arange(len(TOP_K_VALUES))
        width = 0.35
        ax.bar(x - width / 2, base_vals, width, label="Baseline", color="#94a3b8", edgecolor="white")
        ax.bar(x + width / 2, ft_vals, width, label="Fine-tuned", color="#2563eb", edgecolor="white")
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

    # --- 3. Stratified comparison (Hit Rate@10) ---
    buckets = list(base["stratified"].keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    base_hr = [base["stratified"][b].get(f"hit_rate@{hit_k}", 0) for b in buckets]
    ft_hr = [ft["stratified"][b].get(f"hit_rate@{hit_k}", 0) for b in buckets]
    n_queries = [base["stratified"][b].get("n_queries", 0) for b in buckets]
    x = np.arange(len(buckets))
    width = 0.35
    ax.bar(x - width / 2, base_hr, width, label="Baseline", color="#94a3b8", edgecolor="white")
    ax.bar(x + width / 2, ft_hr, width, label="Fine-tuned", color="#2563eb", edgecolor="white")
    ax.set_xlabel("Misconception Frequency Bucket (corpus occurrences)")
    ax.set_ylabel(f"Hit Rate@{hit_k}")
    ax.set_title(f"Stratified Hit Rate@{hit_k}: Does Fine-tuning Help Rare Misconceptions?")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(buckets, n_queries)], fontsize=9)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, f"stratified_comparison_hr{hit_k}.png", FIGURES_DIR)

    # --- 4. t-SNE of corpus embeddings, colored by top misconceptions ---
    print("\n  Generating t-SNE visualization...")
    from sklearn.manifold import TSNE

    base_emb = np.load(EMBEDDINGS_DIR / f"{baseline_tag}_corpus_embeddings.npy")
    ft_emb = np.load(EMBEDDINGS_DIR / f"{finetuned_tag}_corpus_embeddings.npy")

    n_corpus = len(corpus_df)
    rng = np.random.RandomState(RANDOM_SEED)
    tsne_idx = (rng.choice(n_corpus, tsne_max_points, replace=False)
                if n_corpus > tsne_max_points else np.arange(n_corpus))
    tsne_misc = corpus_df.iloc[tsne_idx]["MisconceptionId"].values

    top_10 = [m for m, _ in Counter(tsne_misc).most_common(10)]
    color_map = {m: i for i, m in enumerate(top_10)}
    colors = np.array([color_map.get(m, -1) for m in tsne_misc])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    cmap = matplotlib.colormaps["tab10"]
    for ax, emb, title in zip(axes, [base_emb, ft_emb], ["Baseline", "Fine-tuned"]):
        proj = TSNE(n_components=2, random_state=RANDOM_SEED,
                    perplexity=30, max_iter=1000).fit_transform(emb[tsne_idx])
        other = colors == -1
        ax.scatter(proj[other, 0], proj[other, 1], c="#d1d5db", s=8, alpha=0.3, label="Other")
        for m_id in top_10:
            mask = colors == color_map[m_id]
            ax.scatter(proj[mask, 0], proj[mask, 1], c=[cmap(color_map[m_id])],
                       s=20, alpha=0.7, label=f"Misc {m_id}")
        ax.set_title(f"{title} Embeddings")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, markerscale=2)
    fig.suptitle("t-SNE: Misconception Clusters (Top 10 Misconceptions)", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "tsne_comparison.png", FIGURES_DIR)

    # --- 5. Statistical significance ---
    print("\n  Statistical tests on per-query Hit@%d..." % hit_k)
    query_misc = query_df["MisconceptionId"].values
    corpus_misc = corpus_df["MisconceptionId"].values
    base_hits = per_query_hits(base["indices"], query_misc, corpus_misc, hit_k)
    ft_hits = per_query_hits(ft["indices"], query_misc, corpus_misc, hit_k)

    b, c, mcnemar_p = mcnemar_test(base_hits, ft_hits)
    print(f"    McNemar exact (one-sided, fine-tuned > baseline):")
    print(f"    discordant pairs: baseline-only b={b}, fine-tuned-only c={c}")
    print(f"    p-value: {mcnemar_p:.6f}  "
          f"({'significant' if mcnemar_p < 0.05 else 'not significant'} at α=0.05)")

    if not np.array_equal(base_hits, ft_hits):
        _, wilcoxon_p = scipy_stats.wilcoxon(ft_hits, base_hits, alternative="greater")
    else:
        wilcoxon_p = 1.0
    print(f"    Wilcoxon signed-rank (reference): p={wilcoxon_p:.6f}")

    final = {
        "baseline_metrics": base["metrics"],
        "finetuned_metrics": ft["metrics"],
        "baseline_stratified": base["stratified"],
        "finetuned_stratified": ft["stratified"],
        "mcnemar_discordant_baseline_only": b,
        "mcnemar_discordant_finetuned_only": c,
        "mcnemar_p_value": mcnemar_p,
        "wilcoxon_p_value": float(wilcoxon_p),
        "hit_k": hit_k,
    }
    save_results(final, "final_comparison.json", RESULTS_DIR)
    return final
