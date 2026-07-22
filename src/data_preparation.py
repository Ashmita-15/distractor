"""
Data preparation module for retrieval experiments.

Handles:
1. Question-level stratified train/val/test split
2. Ensuring no data leakage (all QDPs from same question stay together)
3. Reporting split statistics
"""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split

from src.config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED


def create_splits(
    qdp_df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split QDPs into train/val/test sets at the QUESTION level.

    Critical Design Decision:
        Split by QuestionId, not by QDP row. This prevents data leakage
        where different distractors from the same question end up in
        different splits, which would inflate retrieval metrics.

    Strategy:
        1. Get unique QuestionIds
        2. Stratify by the most common MisconceptionId per question
           (best effort — imperfect but better than random)
        3. Split questions into train/val/test
        4. Assign all QDPs from each question to the same split

    Args:
        qdp_df: DataFrame of labeled QDPs.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_qdp, val_qdp, test_qdp) DataFrames.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    # Get the most common misconception per question for stratification
    question_misc = (
        qdp_df.groupby("QuestionId")["MisconceptionId"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
        .reset_index()
        .rename(columns={"MisconceptionId": "stratify_misc"})
    )

    # For stratification, bucket rare misconceptions together
    misc_counts = question_misc["stratify_misc"].value_counts()
    rare_threshold = 3  # Misconceptions appearing in < 3 questions
    rare_misconceptions = set(misc_counts[misc_counts < rare_threshold].index)

    question_misc["stratify_label"] = question_misc["stratify_misc"].apply(
        lambda x: x if x not in rare_misconceptions else -1  # -1 = "rare" bucket
    )

    # Split: first train vs (val+test), then val vs test
    question_ids = question_misc["QuestionId"].values
    stratify_labels = question_misc["stratify_label"].values

    val_test_ratio = val_ratio + test_ratio
    try:
        train_qids, valtest_qids, _, valtest_labels = train_test_split(
            question_ids, stratify_labels,
            test_size=val_test_ratio,
            stratify=stratify_labels,
            random_state=seed,
        )
    except ValueError:
        # Stratification failed (too few samples in some strata)
        print("  ⚠ Stratified split failed, falling back to random split")
        train_qids, valtest_qids = train_test_split(
            question_ids, test_size=val_test_ratio, random_state=seed
        )
        valtest_labels = None

    # Split val+test into val and test
    relative_test_ratio = test_ratio / val_test_ratio
    try:
        if valtest_labels is not None:
            val_qids, test_qids = train_test_split(
                valtest_qids, test_size=relative_test_ratio,
                stratify=valtest_labels,
                random_state=seed,
            )
        else:
            val_qids, test_qids = train_test_split(
                valtest_qids, test_size=relative_test_ratio, random_state=seed
            )
    except ValueError:
        val_qids, test_qids = train_test_split(
            valtest_qids, test_size=relative_test_ratio, random_state=seed
        )

    # Assign QDPs to splits
    train_set = set(train_qids)
    val_set = set(val_qids)
    test_set = set(test_qids)

    train_qdp = qdp_df[qdp_df["QuestionId"].isin(train_set)].reset_index(drop=True)
    val_qdp = qdp_df[qdp_df["QuestionId"].isin(val_set)].reset_index(drop=True)
    test_qdp = qdp_df[qdp_df["QuestionId"].isin(test_set)].reset_index(drop=True)

    # Verify no leakage
    assert len(set(train_qdp["QuestionId"]) & set(val_qdp["QuestionId"])) == 0, \
        "Train-Val leakage detected!"
    assert len(set(train_qdp["QuestionId"]) & set(test_qdp["QuestionId"])) == 0, \
        "Train-Test leakage detected!"
    assert len(set(val_qdp["QuestionId"]) & set(test_qdp["QuestionId"])) == 0, \
        "Val-Test leakage detected!"

    # Report statistics
    _report_split_stats("Train", train_qdp)
    _report_split_stats("Val", val_qdp)
    _report_split_stats("Test", test_qdp)

    # Check misconception coverage
    train_miscs = set(train_qdp["MisconceptionId"].unique())
    val_miscs = set(val_qdp["MisconceptionId"].unique())
    test_miscs = set(test_qdp["MisconceptionId"].unique())

    val_unseen = val_miscs - train_miscs
    test_unseen = test_miscs - train_miscs
    print(f"\n  Misconceptions in val NOT in train: {len(val_unseen)}")
    print(f"  Misconceptions in test NOT in train: {len(test_unseen)}")

    return train_qdp, val_qdp, test_qdp


def _report_split_stats(name: str, df: pd.DataFrame) -> None:
    """Print summary statistics for a data split."""
    print(f"\n  {name} split:")
    print(f"    QDPs:           {len(df):,}")
    print(f"    Questions:      {df['QuestionId'].nunique():,}")
    print(f"    Misconceptions: {df['MisconceptionId'].nunique():,}")
    print(f"    Subjects:       {df['SubjectId'].nunique():,}")
    print(f"    Constructs:     {df['ConstructId'].nunique():,}")
