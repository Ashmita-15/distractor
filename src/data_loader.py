"""
Data loading and preprocessing for the Eedi Misconception dataset.

This module handles:
1. Loading raw CSV files
2. Validating data integrity
3. Melting wide-format questions into question-distractor pairs (QDPs)
4. Merging misconception descriptions

Design Decision:
    The atomic retrieval unit is a Question-Distractor Pair (QDP), not a
    question. A single question can expose up to 3 different misconceptions
    through its distractors. Each QDP is independently retrievable.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional

from src.config import (
    TRAIN_CSV,
    TEST_CSV,
    MISCONCEPTION_CSV,
    SAMPLE_SUBMISSION_CSV,
    QDP_TEXT_TEMPLATE,
)


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all raw dataset files.

    Returns:
        Tuple of (train_df, test_df, misconception_df).
    """
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    misconception_df = pd.read_csv(MISCONCEPTION_CSV)

    print(f"  Train:          {train_df.shape[0]:,} questions, {train_df.shape[1]} columns")
    print(f"  Test:           {test_df.shape[0]:,} questions, {test_df.shape[1]} columns")
    print(f"  Misconceptions: {misconception_df.shape[0]:,} entries")

    return train_df, test_df, misconception_df


def validate_data_integrity(
    train_df: pd.DataFrame,
    misconception_df: pd.DataFrame,
) -> Dict[str, any]:
    """
    Validate dataset integrity.

    Checks:
        1. No duplicate QuestionIds
        2. All misconception IDs in train reference valid IDs in the mapping
        3. Correct answer column has valid values
        4. No missing question text or answer text

    Args:
        train_df: Training dataframe.
        misconception_df: Misconception mapping dataframe.

    Returns:
        Dictionary with validation results.
    """
    results = {}

    # Check 1: No duplicate QuestionIds
    n_duplicates = train_df["QuestionId"].duplicated().sum()
    results["duplicate_question_ids"] = int(n_duplicates)

    # Check 2: Referential integrity of misconception IDs
    valid_ids = set(misconception_df["MisconceptionId"].values)
    invalid_refs = []
    for col in ["MisconceptionAId", "MisconceptionBId", "MisconceptionCId", "MisconceptionDId"]:
        ids_in_train = train_df[col].dropna().astype(int).unique()
        invalid = set(ids_in_train) - valid_ids
        if len(invalid) > 0:
            invalid_refs.append((col, invalid))
    results["invalid_misconception_refs"] = invalid_refs

    # Check 3: Valid correct answer values
    valid_answers = {"A", "B", "C", "D"}
    invalid_answers = set(train_df["CorrectAnswer"].unique()) - valid_answers
    results["invalid_correct_answers"] = invalid_answers

    # Check 4: No missing text fields
    text_cols = [
        "QuestionText", "AnswerAText", "AnswerBText",
        "AnswerCText", "AnswerDText",
    ]
    missing_text = {col: int(train_df[col].isnull().sum()) for col in text_cols}
    results["missing_text_fields"] = missing_text

    # Print summary
    print(f"  Duplicate QuestionIds:       {results['duplicate_question_ids']}")
    print(f"  Invalid misconception refs:  {len(results['invalid_misconception_refs'])}")
    print(f"  Invalid correct answers:     {results['invalid_correct_answers']}")
    print(f"  Missing text fields:         {sum(missing_text.values())}")

    all_valid = (
        n_duplicates == 0
        and len(invalid_refs) == 0
        and len(invalid_answers) == 0
        and sum(missing_text.values()) == 0
    )
    results["all_valid"] = all_valid
    print(f"\n  ✅ All integrity checks passed." if all_valid else "\n  ❌ Some checks failed!")

    return results


def melt_to_qdp(
    train_df: pd.DataFrame,
    misconception_df: pd.DataFrame,
    keep_unlabeled: bool = False,
) -> pd.DataFrame:
    """
    Melt wide-format question data into long-format Question-Distractor Pairs.

    For each question, extracts up to 3 QDPs (one per incorrect option).
    Each QDP contains the question text, distractor text, and misconception label.

    Args:
        train_df: Training dataframe in wide format.
        misconception_df: Misconception mapping for name lookup.
        keep_unlabeled: If True, keep QDPs with NaN misconception IDs.
                       Default False (filter them out for supervised training).

    Returns:
        DataFrame with columns:
            QuestionId, ConstructId, ConstructName, SubjectId, SubjectName,
            QuestionText, CorrectAnswer, CorrectAnswerText, DistractorLetter,
            DistractorText, MisconceptionId, MisconceptionName
    """
    # Map letter -> answer text column
    answer_col_map = {
        "A": "AnswerAText",
        "B": "AnswerBText",
        "C": "AnswerCText",
        "D": "AnswerDText",
    }
    misconception_col_map = {
        "A": "MisconceptionAId",
        "B": "MisconceptionBId",
        "C": "MisconceptionCId",
        "D": "MisconceptionDId",
    }

    rows = []
    for _, q in train_df.iterrows():
        correct_letter = q["CorrectAnswer"]
        correct_text = q[answer_col_map[correct_letter]]

        for letter in ["A", "B", "C", "D"]:
            if letter == correct_letter:
                continue  # Skip the correct answer

            misconception_id = q[misconception_col_map[letter]]
            distractor_text = q[answer_col_map[letter]]

            # Skip if no misconception label and we're filtering
            if not keep_unlabeled and pd.isna(misconception_id):
                continue

            rows.append({
                "QuestionId": int(q["QuestionId"]),
                "ConstructId": int(q["ConstructId"]),
                "ConstructName": q["ConstructName"],
                "SubjectId": int(q["SubjectId"]),
                "SubjectName": q["SubjectName"],
                "QuestionText": q["QuestionText"],
                "CorrectAnswer": correct_letter,
                "CorrectAnswerText": correct_text,
                "DistractorLetter": letter,
                "DistractorText": distractor_text,
                "MisconceptionId": int(misconception_id) if not pd.isna(misconception_id) else None,
            })

    qdp_df = pd.DataFrame(rows)

    # Merge misconception names
    if not keep_unlabeled:
        qdp_df = qdp_df.merge(
            misconception_df,
            on="MisconceptionId",
            how="left",
        )
    else:
        qdp_df = qdp_df.merge(
            misconception_df,
            on="MisconceptionId",
            how="left",
        )

    print(f"  Total QDPs created:  {len(qdp_df):,}")
    if not keep_unlabeled:
        print(f"  (filtered to labeled QDPs only)")
    print(f"  Unique questions:    {qdp_df['QuestionId'].nunique():,}")
    print(f"  Unique misconceptions: {qdp_df['MisconceptionId'].nunique():,}")

    return qdp_df


def create_text_representation(qdp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the text representation for each QDP.

    Uses the template from config to concatenate question context,
    subject/construct metadata, and distractor text into a single
    string suitable for embedding.

    Args:
        qdp_df: DataFrame of question-distractor pairs.

    Returns:
        Same DataFrame with an added 'text' column.
    """
    qdp_df = qdp_df.copy()
    qdp_df["text"] = qdp_df.apply(
        lambda row: QDP_TEXT_TEMPLATE.format(
            subject=row["SubjectName"],
            construct=row["ConstructName"],
            question=row["QuestionText"],
            correct_answer=row["CorrectAnswerText"],
            distractor=row["DistractorText"],
        ),
        axis=1,
    )
    # Report text length statistics
    lengths = qdp_df["text"].str.len()
    print(f"  Text length — min: {lengths.min()}, median: {lengths.median():.0f}, "
          f"max: {lengths.max()}, mean: {lengths.mean():.0f}")

    return qdp_df


def get_misconception_stats(qdp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-misconception statistics.

    Args:
        qdp_df: DataFrame of labeled QDPs.

    Returns:
        DataFrame indexed by MisconceptionId with count, name, and
        number of unique constructs/subjects the misconception spans.
    """
    stats = (
        qdp_df.groupby("MisconceptionId")
        .agg(
            count=("QuestionId", "size"),
            n_unique_questions=("QuestionId", "nunique"),
            n_unique_constructs=("ConstructId", "nunique"),
            n_unique_subjects=("SubjectId", "nunique"),
            misconception_name=("MisconceptionName", "first"),
        )
        .sort_values("count", ascending=False)
        .reset_index()
    )
    return stats
