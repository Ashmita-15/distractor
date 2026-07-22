"""
Triplet construction for misconception-aware fine-tuning.

Theory:
    Triplet loss trains embeddings so that an anchor is closer to a positive
    (same misconception) than a negative (different misconception) by at
    least a margin α:
        L = max(0, ||f(a) - f(p)||² - ||f(a) - f(n)||² + α)

    The quality of triplets directly affects learning:
    - Too-easy negatives → uninformative gradients (loss is already 0)
    - Too-hard negatives → noisy gradients
    - Optimal: "semi-hard" negatives that are close to the anchor but wrong

Negative Sampling Strategies:
    1. Random: Any QDP with a different misconception
    2. In-subject hard: Same subject, different misconception (recommended)
    3. Semi-hard mining: Requires initial embeddings (used in later epochs)
"""

import json
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from collections import defaultdict

from src.config import RANDOM_SEED, TRIPLETS_DIR


def save_triplets(
    triplets: List[Tuple[str, str, str]],
    filename: str,
    output_dir: Path = TRIPLETS_DIR,
) -> Path:
    """
    Save triplets as JSONL (one {"anchor", "positive", "negative"} per line).

    JSONL keeps the file human-inspectable and streams cleanly on any
    platform, which matters for uploading to Colab/Kaggle.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, "w") as f:
        for a, p, n in triplets:
            f.write(json.dumps({"anchor": a, "positive": p, "negative": n}) + "\n")
    print(f"  [Saved] {path} ({len(triplets):,} triplets)")
    return path


def load_triplets(
    filename: str,
    triplets_dir: Path = TRIPLETS_DIR,
) -> List[Tuple[str, str, str]]:
    """Load triplets saved by save_triplets."""
    path = Path(triplets_dir) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Triplet file not found: {path}. "
            f"Run `python scripts/03_create_triplets.py` first."
        )
    triplets = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            triplets.append((d["anchor"], d["positive"], d["negative"]))
    print(f"  [Loaded] {path} ({len(triplets):,} triplets)")
    return triplets


def package_triplets(
    filenames: List[str],
    archive_name: str = "triplets_package.zip",
    triplets_dir: Path = TRIPLETS_DIR,
) -> Path:
    """
    Zip triplet files into a single archive for upload to Colab/Kaggle.
    """
    archive_path = triplets_dir / archive_name
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in filenames:
            zf.write(triplets_dir / fn, arcname=fn)
    print(f"  [Saved] {archive_path}")
    return archive_path


def construct_triplets(
    qdp_df: pd.DataFrame,
    strategy: str = "in_subject",
    negatives_per_anchor: int = 3,
    seed: int = RANDOM_SEED,
    min_positives: int = 2,
) -> List[Tuple[str, str, str]]:
    """
    Construct (anchor, positive, negative) triplets from labeled QDPs.

    Args:
        qdp_df: DataFrame with 'text', 'MisconceptionId', 'QuestionId', 'SubjectId'.
        strategy: Negative sampling strategy — 'random' or 'in_subject'.
        negatives_per_anchor: Number of negative samples per anchor-positive pair.
        seed: Random seed.
        min_positives: Minimum number of QDPs a misconception must have to form triplets.

    Returns:
        List of (anchor_text, positive_text, negative_text) tuples.
    """
    rng = np.random.RandomState(seed)

    # Group QDPs by misconception
    misc_groups = defaultdict(list)
    for idx, row in qdp_df.iterrows():
        misc_groups[row["MisconceptionId"]].append(idx)

    # Filter misconceptions with too few examples
    viable_misconceptions = {
        m: indices for m, indices in misc_groups.items()
        if len(indices) >= min_positives
    }
    excluded_count = len(misc_groups) - len(viable_misconceptions)

    print(f"  Total misconceptions: {len(misc_groups)}")
    print(f"  Viable (count >= {min_positives}): {len(viable_misconceptions)}")
    print(f"  Excluded (singletons): {excluded_count}")

    # Group by subject for in-subject negative sampling
    subject_groups = defaultdict(list)
    for idx, row in qdp_df.iterrows():
        subject_groups[row["SubjectId"]].append(idx)

    # Build misconception-to-indices lookup for efficient exclusion
    all_indices = set(qdp_df.index)
    misc_index_sets = {m: set(indices) for m, indices in misc_groups.items()}

    triplets = []

    for misc_id, anchor_indices in viable_misconceptions.items():
        for i, anchor_idx in enumerate(anchor_indices):
            anchor_row = qdp_df.loc[anchor_idx]
            anchor_text = anchor_row["text"]
            anchor_qid = anchor_row["QuestionId"]
            anchor_subject = anchor_row["SubjectId"]

            # Positive: different QDP with same misconception, different question
            positive_candidates = [
                idx for idx in anchor_indices
                if idx != anchor_idx and qdp_df.loc[idx]["QuestionId"] != anchor_qid
            ]

            if len(positive_candidates) == 0:
                continue  # No valid positive available

            # Select a positive
            pos_idx = rng.choice(positive_candidates)
            positive_text = qdp_df.loc[pos_idx]["text"]

            # Negative sampling
            if strategy == "in_subject":
                # Same subject, different misconception
                subject_candidates = [
                    idx for idx in subject_groups[anchor_subject]
                    if idx not in misc_index_sets[misc_id]
                    and qdp_df.loc[idx]["QuestionId"] != anchor_qid
                ]
                # Fallback to random if not enough in-subject negatives
                if len(subject_candidates) < negatives_per_anchor:
                    neg_pool = [
                        idx for idx in all_indices
                        if idx not in misc_index_sets[misc_id]
                        and qdp_df.loc[idx]["QuestionId"] != anchor_qid
                    ]
                else:
                    neg_pool = subject_candidates
            else:
                # Random negatives
                neg_pool = [
                    idx for idx in all_indices
                    if idx not in misc_index_sets[misc_id]
                    and qdp_df.loc[idx]["QuestionId"] != anchor_qid
                ]

            if len(neg_pool) == 0:
                continue

            n_neg = min(negatives_per_anchor, len(neg_pool))
            neg_indices = rng.choice(neg_pool, size=n_neg, replace=False)

            for neg_idx in neg_indices:
                negative_text = qdp_df.loc[neg_idx]["text"]
                triplets.append((anchor_text, positive_text, negative_text))

    print(f"  Total triplets constructed: {len(triplets):,}")
    print(f"  Strategy: {strategy}")

    return triplets


def validate_triplets(
    triplets: List[Tuple[str, str, str]],
    n_samples: int = 5,
) -> None:
    """
    Print sample triplets for manual validation.

    Args:
        triplets: List of (anchor, positive, negative) text tuples.
        n_samples: Number of samples to print.
    """
    print(f"\n  === Sample Triplets (showing {n_samples}) ===\n")
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(triplets), size=min(n_samples, len(triplets)), replace=False)

    for i, idx in enumerate(sample_indices):
        anchor, positive, negative = triplets[idx]
        print(f"  --- Triplet {i+1} ---")
        print(f"  Anchor:   {anchor[:120]}...")
        print(f"  Positive: {positive[:120]}...")
        print(f"  Negative: {negative[:120]}...")
        print()


def triplets_to_input_examples(
    triplets: List[Tuple[str, str, str]],
):
    """
    Convert triplets to SentenceTransformer InputExample format.

    Args:
        triplets: List of (anchor, positive, negative) text tuples.

    Returns:
        List of InputExample objects for training.
    """
    from sentence_transformers import InputExample

    examples = []
    for anchor, positive, negative in triplets:
        examples.append(InputExample(texts=[anchor, positive, negative]))
    return examples
