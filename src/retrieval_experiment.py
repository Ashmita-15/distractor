"""
Shared retrieval-evaluation logic used by all evaluation entry points.

A "retrieval experiment" is fully specified by:
    - an embedding model (HuggingFace name or local fine-tuned path)
    - a corpus (train+val QDPs under the standard protocol)
    - a query set (test QDPs under the standard protocol)

This module encodes both sides, performs exact top-K retrieval, computes
overall and stratified metrics, and persists every artifact needed for a
later model-vs-model comparison (embeddings, retrieved indices, metrics)
without re-encoding.

Keeping this in one place guarantees the baseline and the fine-tuned model
are evaluated by byte-identical code paths.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.config import (
    RESULTS_DIR,
    EMBEDDINGS_DIR,
    TOP_K_VALUES,
    MAX_K,
)
from src.utils import save_results
from src.baseline_retrieval import (
    encode_texts,
    build_faiss_index,
    retrieve_top_k,
    save_embeddings,
)
from src.evaluation import compute_retrieval_metrics, compute_stratified_metrics


def load_split(name: str, results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """
    Load a prepared QDP split saved by scripts/01_prepare_dataset.py.

    Args:
        name: One of 'train', 'val', 'test'.
        results_dir: Directory containing the split CSVs.

    Returns:
        DataFrame of QDPs.

    Raises:
        FileNotFoundError with a actionable message if the split is missing.
    """
    path = results_dir / f"{name}_qdp.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}. "
            f"Run `python scripts/01_prepare_dataset.py` first."
        )
    return pd.read_csv(path)


def load_corpus_and_queries(
    query_split: str = "test",
) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Assemble the retrieval corpus and query set under the standard protocol.

    Protocol:
        Corpus = train + val QDPs. Queries = test QDPs (disjoint from corpus).
        If query_split='val', queries are corpus members, and the returned
        queries_in_corpus flag is True so that evaluation excludes
        self-retrieval from the relevant counts.

    Args:
        query_split: 'test' (default, standard protocol) or 'val'.

    Returns:
        Tuple of (corpus_df, query_df, queries_in_corpus).
    """
    if query_split not in {"test", "val"}:
        raise ValueError(f"query_split must be 'test' or 'val', got {query_split!r}")

    train_qdp = load_split("train")
    val_qdp = load_split("val")
    corpus_df = pd.concat([train_qdp, val_qdp], ignore_index=True)

    if query_split == "test":
        return corpus_df, load_split("test"), False
    return corpus_df, val_qdp.reset_index(drop=True), True


def run_retrieval_evaluation(
    model_name_or_path: str,
    tag: str,
    query_split: str = "test",
    results_dir: Path = RESULTS_DIR,
    embeddings_dir: Path = EMBEDDINGS_DIR,
    k_values=TOP_K_VALUES,
    max_k: int = MAX_K,
    query_template: Optional[str] = None,
    corpus_template: Optional[str] = None,
    corpus_embeddings: Optional[np.ndarray] = None,
) -> Dict:
    """
    Run the complete retrieval evaluation for one embedding model.

    Saves (all prefixed by `tag`):
        {embeddings_dir}/{tag}_corpus_embeddings.npy
        {embeddings_dir}/{tag}_query_embeddings.npy
        {results_dir}/{tag}_indices.npy      — retrieved corpus indices (n_queries, max_k)
        {results_dir}/{tag}_metrics.json
        {results_dir}/{tag}_stratified_metrics.json

    Args:
        model_name_or_path: SentenceTransformer name or local model path.
        tag: Artifact prefix, e.g. 'baseline' or 'finetuned'.
        query_split: 'test' (standard) or 'val'.
        results_dir: Where metric/index artifacts are written.
        embeddings_dir: Where embeddings are written.
        k_values: K values for metric computation.
        max_k: Number of neighbors to retrieve.
        query_template: Optional override for the QUERY text representation
            (query-representation ablation). None = use the prepared 'text'
            column, i.e. the canonical representation used by all prior
            experiments.
        corpus_template: Optional override for the CORPUS text representation.
            None = use the prepared 'text' column. Deployment-faithful
            ablations leave this as None, since historical corpus items do
            have known distractors.
        corpus_embeddings: Optional precomputed corpus embeddings. Supplying
            these guarantees byte-identical corpus representation across
            several query variants (and avoids redundant encoding). Must
            correspond to the corpus produced by load_corpus_and_queries.

    Returns:
        Dict with keys: metrics, stratified, indices, corpus_df, query_df.
    """
    from src.data_loader import create_text_representation

    corpus_df, query_df, queries_in_corpus = load_corpus_and_queries(query_split)
    print(f"  Corpus size: {len(corpus_df):,} QDPs")
    print(f"  Query size:  {len(query_df):,} QDPs (split: {query_split})")

    if corpus_template is not None:
        corpus_df = create_text_representation(corpus_df, template=corpus_template)
        print(f"  Corpus template overridden: {corpus_template!r}")
    if query_template is not None:
        query_df = create_text_representation(query_df, template=query_template)
        print(f"  Query template overridden:  {query_template!r}")
        print(f"  Example query: {query_df['text'].iloc[0][:160]!r}")

    if corpus_embeddings is not None:
        if len(corpus_embeddings) != len(corpus_df):
            raise ValueError(
                f"corpus_embeddings has {len(corpus_embeddings)} rows but corpus "
                f"has {len(corpus_df)}; they must correspond."
            )
        print("\nReusing precomputed corpus embeddings (identical across variants).")
        corpus_emb = corpus_embeddings
    else:
        print("\nEncoding corpus...")
        corpus_emb = encode_texts(corpus_df["text"].tolist(), model_name_or_path=str(model_name_or_path))
    save_embeddings(corpus_emb, f"{tag}_corpus_embeddings.npy", embeddings_dir)

    print("\nEncoding queries...")
    query_emb = encode_texts(query_df["text"].tolist(), model_name_or_path=str(model_name_or_path))
    save_embeddings(query_emb, f"{tag}_query_embeddings.npy", embeddings_dir)

    print("\nBuilding index and retrieving...")
    index = build_faiss_index(corpus_emb)
    # When queries are corpus members (val split), exclusion of the query
    # itself and of same-question QDPs is handled by retrieve_top_k.
    scores, indices = retrieve_top_k(
        query_embeddings=query_emb,
        index=index,
        k=max_k,
        query_question_ids=query_df["QuestionId"].values,
        corpus_question_ids=corpus_df["QuestionId"].values,
    )
    np.save(results_dir / f"{tag}_indices.npy", indices)
    print(f"  [Saved] {results_dir / f'{tag}_indices.npy'}")

    query_misc = query_df["MisconceptionId"].values
    corpus_misc = corpus_df["MisconceptionId"].values
    retrieved_misc = corpus_misc[indices]

    print("\nComputing metrics...")
    metrics = compute_retrieval_metrics(
        query_misconception_ids=query_misc,
        retrieved_misconception_ids=retrieved_misc,
        corpus_misconception_ids=corpus_misc,
        k_values=k_values,
        queries_in_corpus=queries_in_corpus,
    )
    stratified = compute_stratified_metrics(
        query_misconception_ids=query_misc,
        retrieved_misconception_ids=retrieved_misc,
        corpus_misconception_ids=corpus_misc,
        k_values=k_values,
    )

    save_results(metrics, f"{tag}_metrics.json", results_dir)
    save_results(stratified, f"{tag}_stratified_metrics.json", results_dir)

    return {
        "metrics": metrics,
        "stratified": stratified,
        "indices": indices,
        "corpus_df": corpus_df,
        "query_df": query_df,
    }


def print_metrics_summary(metrics: Dict, tag: str, k_values=TOP_K_VALUES) -> None:
    """Print a compact metrics table for console output."""
    print(f"\n  === {tag} Retrieval Metrics ===")
    print(f"  Queries: {metrics['n_queries']} "
          f"(attainable: {metrics['n_attainable']}, {metrics['attainable_fraction']:.1%})")
    header = f"  {'K':>3s}  {'HitRate':>8s} {'Recall':>8s} {'MRR':>8s} {'nDCG':>8s} {'HR(att.)':>9s}"
    print(header)
    for k in k_values:
        print(f"  {k:3d}  {metrics[f'hit_rate@{k}']:8.4f} {metrics[f'recall@{k}']:8.4f} "
              f"{metrics[f'mrr@{k}']:8.4f} {metrics[f'ndcg@{k}']:8.4f} "
              f"{metrics[f'hit_rate_attainable@{k}']:9.4f}")
