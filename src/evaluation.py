"""
Retrieval evaluation metrics for misconception-aware retrieval.

Implements standard IR metrics adapted for our task:
- A retrieved QDP is "relevant" if it shares the same MisconceptionId
  as the query QDP.

Metrics:
    - Hit Rate@K: Binary — does at least one relevant item appear in top K?
    - Recall@K: What fraction of all relevant items appear in top K?
    - MRR: Mean Reciprocal Rank — average of 1/rank_of_first_relevant
    - nDCG@K: Normalized Discounted Cumulative Gain (binary relevance)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from collections import defaultdict

from src.config import TOP_K_VALUES


def compute_retrieval_metrics(
    query_misconception_ids: np.ndarray,
    retrieved_misconception_ids: np.ndarray,
    corpus_misconception_ids: np.ndarray,
    k_values: List[int] = TOP_K_VALUES,
    query_question_ids: Optional[np.ndarray] = None,
    retrieved_question_ids: Optional[np.ndarray] = None,
    queries_in_corpus: bool = False,
) -> Dict[str, float]:
    """
    Compute retrieval metrics across all queries.

    Note on Recall@K denominators:
        In this project's evaluation protocol the query set (test split) is
        DISJOINT from the retrieval corpus (train+val), so every corpus item
        sharing the query's misconception is retrievable and counts as
        relevant. Set queries_in_corpus=True only if queries are corpus
        members, in which case one occurrence (the query itself) is excluded
        from the relevant count.

    Args:
        query_misconception_ids: Shape (n_queries,) — misconception ID of each query.
        retrieved_misconception_ids: Shape (n_queries, max_k) — misconception IDs
            of retrieved items for each query.
        corpus_misconception_ids: Shape (n_corpus,) — misconception IDs of all
            items in the retrieval corpus (for computing total relevant count).
        k_values: List of K values to evaluate at.
        query_question_ids: Optional, shape (n_queries,) — for excluding
            same-question results if not already done.
        retrieved_question_ids: Optional, shape (n_queries, max_k) — question IDs
            of retrieved items.

    Returns:
        Dictionary of metric_name -> value, e.g. {"hit_rate@5": 0.42, ...}
    """
    n_queries = len(query_misconception_ids)
    max_k = max(k_values)

    # Precompute total relevant items per misconception in the corpus
    from collections import Counter
    corpus_misc_counts = Counter(corpus_misconception_ids.tolist())

    results = {}

    # Attainability: a query is attainable only if its misconception appears
    # at least once in the corpus. Unattainable queries have a hard Hit Rate
    # of 0 regardless of model quality, so the attainable fraction is the
    # ceiling on Hit Rate@K. We report overall metrics (all queries) AND
    # metrics restricted to attainable queries.
    attainable_mask = np.array([
        corpus_misc_counts.get(m, 0) > 0 for m in query_misconception_ids
    ])
    results["n_queries"] = int(n_queries)
    results["n_attainable"] = int(attainable_mask.sum())
    results["attainable_fraction"] = float(attainable_mask.mean()) if n_queries > 0 else 0.0

    for k in k_values:
        hit_rates = []
        recalls = []
        reciprocal_ranks = []
        ndcgs = []

        for i in range(n_queries):
            query_misc = query_misconception_ids[i]
            retrieved = retrieved_misconception_ids[i, :k]

            # Binary relevance: does retrieved item share the same misconception?
            relevance = (retrieved == query_misc).astype(float)

            # Hit Rate: at least one relevant in top-K
            hit = 1.0 if relevance.sum() > 0 else 0.0
            hit_rates.append(hit)

            # Recall@K: fraction of all relevant items found
            total_relevant = corpus_misc_counts.get(query_misc, 0)
            if queries_in_corpus:
                total_relevant -= 1  # exclude the query itself
            if total_relevant > 0:
                recall = relevance.sum() / total_relevant
            else:
                recall = 0.0  # Singleton misconception, cannot be retrieved
            recalls.append(recall)

            # MRR: reciprocal rank of first relevant item
            relevant_positions = np.where(relevance > 0)[0]
            if len(relevant_positions) > 0:
                rr = 1.0 / (relevant_positions[0] + 1)
            else:
                rr = 0.0
            reciprocal_ranks.append(rr)

            # nDCG@K with binary relevance
            dcg = np.sum(relevance / np.log2(np.arange(2, k + 2)))
            # Ideal DCG: all relevant items ranked first
            ideal_relevant_count = min(int(total_relevant), k) if total_relevant > 0 else 0
            ideal_relevance = np.zeros(k)
            ideal_relevance[:ideal_relevant_count] = 1.0
            idcg = np.sum(ideal_relevance / np.log2(np.arange(2, k + 2)))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)

        hit_rates = np.array(hit_rates)
        recalls = np.array(recalls)
        reciprocal_ranks = np.array(reciprocal_ranks)
        ndcgs = np.array(ndcgs)

        results[f"hit_rate@{k}"] = float(hit_rates.mean())
        results[f"recall@{k}"] = float(recalls.mean())
        results[f"mrr@{k}"] = float(reciprocal_ranks.mean())
        results[f"ndcg@{k}"] = float(ndcgs.mean())

        # Metrics over attainable queries only (removes the fixed ceiling
        # imposed by corpus singleton/unseen misconceptions)
        if attainable_mask.sum() > 0:
            results[f"hit_rate_attainable@{k}"] = float(hit_rates[attainable_mask].mean())
            results[f"mrr_attainable@{k}"] = float(reciprocal_ranks[attainable_mask].mean())
        else:
            results[f"hit_rate_attainable@{k}"] = 0.0
            results[f"mrr_attainable@{k}"] = 0.0

    return results


def compute_stratified_metrics(
    query_misconception_ids: np.ndarray,
    retrieved_misconception_ids: np.ndarray,
    corpus_misconception_ids: np.ndarray,
    k_values: List[int] = TOP_K_VALUES,
    frequency_buckets: Optional[Dict[str, set]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics stratified by misconception frequency bucket.

    This reveals whether the model disproportionately fails on rare
    misconceptions — a critical analysis for the long-tail problem.

    Args:
        query_misconception_ids: Misconception IDs of query QDPs.
        retrieved_misconception_ids: Misconception IDs of retrieved items.
        corpus_misconception_ids: All misconception IDs in corpus.
        k_values: K values for evaluation.
        frequency_buckets: Dict of bucket_name -> set of MisconceptionIds.
            If None, auto-creates buckets from corpus frequency.

    Returns:
        Dict of bucket_name -> metrics_dict.
    """
    from collections import Counter
    corpus_counts = Counter(corpus_misconception_ids.tolist())

    if frequency_buckets is None:
        # "unseen" = query misconceptions absent from the corpus entirely;
        # these queries are unattainable for any retriever and would
        # otherwise be silently dropped from the stratified analysis.
        query_miscs = set(query_misconception_ids.tolist())
        frequency_buckets = {
            "unseen (=0)": {m for m in query_miscs if corpus_counts.get(m, 0) == 0},
            "singleton (=1)": {m for m, c in corpus_counts.items() if c == 1},
            "rare (2-5)": {m for m, c in corpus_counts.items() if 2 <= c <= 5},
            "moderate (6-20)": {m for m, c in corpus_counts.items() if 6 <= c <= 20},
            "frequent (>20)": {m for m, c in corpus_counts.items() if c > 20},
        }

    stratified_results = {}
    for bucket_name, misc_set in frequency_buckets.items():
        # Filter queries belonging to this bucket
        mask = np.array([m in misc_set for m in query_misconception_ids])
        if mask.sum() == 0:
            stratified_results[bucket_name] = {"n_queries": 0}
            continue

        bucket_metrics = compute_retrieval_metrics(
            query_misconception_ids=query_misconception_ids[mask],
            retrieved_misconception_ids=retrieved_misconception_ids[mask],
            corpus_misconception_ids=corpus_misconception_ids,
            k_values=k_values,
        )
        bucket_metrics["n_queries"] = int(mask.sum())
        stratified_results[bucket_name] = bucket_metrics

    return stratified_results


def format_metrics_table(
    metrics: Dict[str, float],
    k_values: List[int] = TOP_K_VALUES,
    model_name: str = "Model",
) -> pd.DataFrame:
    """
    Format metrics into a readable table for display and reporting.

    Args:
        metrics: Dictionary of metric_name -> value.
        k_values: K values used.
        model_name: Name for the model column.

    Returns:
        DataFrame with metrics organized by K.
    """
    rows = []
    for k in k_values:
        rows.append({
            "K": k,
            f"Hit Rate@K": f"{metrics.get(f'hit_rate@{k}', 0):.4f}",
            f"Recall@K": f"{metrics.get(f'recall@{k}', 0):.4f}",
            f"MRR@K": f"{metrics.get(f'mrr@{k}', 0):.4f}",
            f"nDCG@K": f"{metrics.get(f'ndcg@{k}', 0):.4f}",
        })
    return pd.DataFrame(rows)


def compare_models(
    baseline_metrics: Dict[str, float],
    finetuned_metrics: Dict[str, float],
    k_values: List[int] = TOP_K_VALUES,
) -> pd.DataFrame:
    """
    Create a side-by-side comparison table of baseline vs fine-tuned metrics.

    Args:
        baseline_metrics: Metrics from baseline retrieval.
        finetuned_metrics: Metrics from fine-tuned retrieval.
        k_values: K values to compare.

    Returns:
        DataFrame with columns for both models and relative improvement.
    """
    rows = []
    for k in k_values:
        for metric_name in ["hit_rate", "recall", "mrr", "ndcg"]:
            key = f"{metric_name}@{k}"
            baseline_val = baseline_metrics.get(key, 0)
            finetuned_val = finetuned_metrics.get(key, 0)
            if baseline_val > 0:
                improvement = ((finetuned_val - baseline_val) / baseline_val) * 100
            else:
                improvement = float("inf") if finetuned_val > 0 else 0.0

            rows.append({
                "Metric": key,
                "Baseline": f"{baseline_val:.4f}",
                "Fine-tuned": f"{finetuned_val:.4f}",
                "Δ (abs)": f"{finetuned_val - baseline_val:+.4f}",
                "Δ (%)": f"{improvement:+.1f}%",
            })
    return pd.DataFrame(rows)
