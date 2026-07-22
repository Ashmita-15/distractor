"""
Baseline semantic retrieval using pretrained SentenceTransformer and FAISS.

Theory:
    The pretrained all-MiniLM-L6-v2 maps text to 384-dimensional dense vectors
    in a space where cosine similarity correlates with semantic similarity.
    FAISS enables efficient nearest-neighbor search over these embeddings.

    This baseline captures textual/topical similarity but has no explicit
    training signal for pedagogical (misconception-based) similarity.
"""

import sys
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Optional, List
from sentence_transformers import SentenceTransformer

# On macOS, faiss-cpu and torch each bundle their own OpenMP runtime; once
# torch has run compute, any faiss search segfaults (observed empirically —
# neither import ordering, faiss.omp_set_num_threads(1), nor
# KMP_DUPLICATE_LIB_OK avoids it). Exact inner-product search over a corpus
# of ~4K vectors is trivial in NumPy and numerically identical to
# faiss.IndexFlatIP, so we default to the NumPy backend on macOS and keep
# FAISS for platforms where the runtimes coexist.
USE_FAISS_DEFAULT = sys.platform != "darwin"
if USE_FAISS_DEFAULT:
    import faiss


class ExactInnerProductIndex:
    """
    Drop-in replacement for faiss.IndexFlatIP using exact NumPy search.

    Implements the subset of the FAISS Index API used in this project
    (`add`, `search`, `ntotal`). Results are identical to IndexFlatIP:
    both perform exhaustive exact maximum-inner-product search.
    """

    def __init__(self, d: int):
        self.d = d
        self._vectors = np.empty((0, d), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return self._vectors.shape[0]

    def add(self, vectors: np.ndarray) -> None:
        """Add vectors of shape (n, d) to the index."""
        self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exact top-k inner-product search.

        Returns (scores, indices), each of shape (n_queries, k),
        matching the faiss.Index.search return convention.
        """
        k = min(k, self.ntotal)
        sims = queries.astype(np.float32) @ self._vectors.T  # (n_queries, n_corpus)
        # argpartition for top-k, then sort those k by score descending
        top_k = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(sims.shape[0])[:, None]
        order = np.argsort(-sims[rows, top_k], axis=1)
        indices = top_k[rows, order]
        scores = sims[rows, indices]
        return scores.astype(np.float32), indices.astype(np.int64)

from src.config import (
    BASELINE_MODEL_NAME,
    EMBEDDING_DIM,
    MAX_K,
    EMBEDDINGS_DIR,
)
from src.utils import get_device, set_seed


def encode_texts(
    texts: List[str],
    model_name_or_path: str = BASELINE_MODEL_NAME,
    batch_size: int = 64,
    show_progress: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Encode a list of texts into dense embeddings using SentenceTransformer.

    Args:
        texts: List of text strings to encode.
        model_name_or_path: Model name (HuggingFace) or local path.
        batch_size: Encoding batch size.
        show_progress: Whether to show progress bar.
        normalize: Whether to L2-normalize embeddings (required for cosine sim via inner product).

    Returns:
        numpy array of shape (len(texts), embedding_dim).
    """
    set_seed()
    device = get_device()
    print(f"  Loading model: {model_name_or_path}")
    print(f"  Device: {device}")

    model = SentenceTransformer(model_name_or_path, device=str(device))

    print(f"  Encoding {len(texts):,} texts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )

    print(f"  Embeddings shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(
    embeddings: np.ndarray,
    use_inner_product: bool = True,
    use_faiss: bool = USE_FAISS_DEFAULT,
):
    """
    Build an exact inner-product index for nearest-neighbor search.

    Design Decision:
        Inner product over L2-normalized embeddings is equivalent to cosine
        similarity. This is exact search — no approximation — suitable for
        our corpus size (~4K vectors). On macOS the NumPy backend is used
        (identical results) because faiss-cpu's OpenMP runtime clashes with
        torch's; on other platforms faiss.IndexFlatIP is used.

    Args:
        embeddings: Shape (n, d) numpy array of embeddings.
        use_inner_product: If True, use inner product (cosine sim with normalized vectors).
        use_faiss: If True, use faiss.IndexFlatIP; otherwise the NumPy backend.

    Returns:
        Index object exposing add/search/ntotal (faiss.Index or ExactInnerProductIndex).
    """
    d = embeddings.shape[1]
    if use_faiss:
        index = faiss.IndexFlatIP(d) if use_inner_product else faiss.IndexFlatL2(d)
        backend = "FAISS"
    else:
        index = ExactInnerProductIndex(d)
        backend = "NumPy (exact, FAISS-equivalent)"

    # Ensure float32
    embeddings = embeddings.astype(np.float32)
    index.add(embeddings)

    print(f"  {backend} index built: {index.ntotal} vectors, dim={d}")
    return index


def retrieve_top_k(
    query_embeddings: np.ndarray,
    index,
    k: int = MAX_K,
    query_indices: Optional[np.ndarray] = None,
    query_question_ids: Optional[np.ndarray] = None,
    corpus_question_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve top-K nearest neighbors for each query.

    Self-exclusion: If query is part of the corpus, exclude it from results.
    Same-question exclusion: Exclude other QDPs from the same question.

    Args:
        query_embeddings: Shape (n_queries, d) embeddings.
        index: FAISS index built on the corpus.
        k: Number of neighbors to retrieve.
        query_indices: If queries are in the corpus, their corpus indices.
        query_question_ids: QuestionIds of queries (for same-question exclusion).
        corpus_question_ids: QuestionIds of all corpus items.

    Returns:
        Tuple of (scores, indices) each of shape (n_queries, k).
        indices are into the original corpus.
    """
    # Retrieve extra candidates to account for exclusions
    extra = 10  # Buffer for self + same-question exclusion
    fetch_k = k + extra

    query_embeddings = query_embeddings.astype(np.float32)
    raw_scores, raw_indices = index.search(query_embeddings, fetch_k)

    n_queries = len(query_embeddings)
    final_scores = np.zeros((n_queries, k), dtype=np.float32)
    final_indices = np.zeros((n_queries, k), dtype=np.int64)

    for i in range(n_queries):
        count = 0
        for j in range(fetch_k):
            idx = raw_indices[i, j]

            # Self-exclusion
            if query_indices is not None and idx == query_indices[i]:
                continue

            # Same-question exclusion
            if (query_question_ids is not None and corpus_question_ids is not None
                    and corpus_question_ids[idx] == query_question_ids[i]):
                continue

            final_scores[i, count] = raw_scores[i, j]
            final_indices[i, count] = idx
            count += 1

            if count >= k:
                break

    return final_scores, final_indices


def save_embeddings(
    embeddings: np.ndarray,
    filename: str,
    output_dir: Path = EMBEDDINGS_DIR,
) -> Path:
    """Save embeddings to disk as numpy array."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    np.save(filepath, embeddings)
    print(f"  [Saved] {filepath}")
    return filepath


def load_embeddings(
    filename: str,
    output_dir: Path = EMBEDDINGS_DIR,
) -> np.ndarray:
    """Load embeddings from disk."""
    filepath = output_dir / filename
    embeddings = np.load(filepath)
    print(f"  [Loaded] {filepath} — shape: {embeddings.shape}")
    return embeddings
