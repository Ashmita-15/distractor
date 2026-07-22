"""
Research-oriented Exploratory Data Analysis for the Eedi dataset.

Generates publication-quality figures and statistics that inform
modeling decisions (class imbalance, text properties, co-occurrence).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter
from typing import Optional

from src.config import FIGURES_DIR
from src.utils import setup_matplotlib, save_figure


def run_full_eda(
    train_df: pd.DataFrame,
    qdp_df: pd.DataFrame,
    misconception_df: pd.DataFrame,
    save_dir: Optional[str] = None,
) -> dict:
    """
    Run the complete EDA pipeline and generate all figures.

    Args:
        train_df: Raw training dataframe (question-level).
        qdp_df: Question-distractor pairs (long format, labeled).
        misconception_df: Misconception mapping table.
        save_dir: Directory to save figures. Defaults to FIGURES_DIR.

    Returns:
        Dictionary of summary statistics.
    """
    setup_matplotlib()
    output_dir = save_dir or FIGURES_DIR
    stats = {}

    # 1. Misconception frequency distribution
    print("  [1/7] Misconception frequency distribution...")
    stats["misconception_freq"] = plot_misconception_frequency(qdp_df, output_dir)

    # 2. Subject distribution
    print("  [2/7] Subject distribution...")
    stats["subject_dist"] = plot_subject_distribution(train_df, output_dir)

    # 3. Construct distribution
    print("  [3/7] Construct distribution...")
    stats["construct_dist"] = plot_construct_distribution(train_df, output_dir)

    # 4. Misconception-construct co-occurrence
    print("  [4/7] Misconception-construct co-occurrence...")
    stats["cooccurrence"] = analyze_misconception_construct_cooccurrence(qdp_df, output_dir)

    # 5. Question text analysis
    print("  [5/7] Question text analysis...")
    stats["text_analysis"] = analyze_question_text(train_df, output_dir)

    # 6. Misconception label coverage
    print("  [6/7] Misconception label coverage...")
    stats["label_coverage"] = analyze_label_coverage(train_df, output_dir)

    # 7. Distractor-per-question distribution
    print("  [7/7] Distractors per question...")
    stats["qdp_per_question"] = analyze_qdp_per_question(qdp_df, output_dir)

    return stats


def plot_misconception_frequency(qdp_df: pd.DataFrame, output_dir) -> dict:
    """
    Plot the frequency distribution of misconceptions (Zipf-like analysis).

    This reveals the long-tail nature of the data which directly impacts
    triplet construction (Phase 6) — rare misconceptions have few positives.
    """
    freq = qdp_df["MisconceptionId"].value_counts().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Rank-frequency plot (log-log for Zipf analysis)
    ax = axes[0]
    ranks = np.arange(1, len(freq) + 1)
    ax.plot(ranks, freq.values, color="#2563eb", linewidth=1.5, alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Misconception Rank (log scale)")
    ax.set_ylabel("Frequency (log scale)")
    ax.set_title("Misconception Rank-Frequency Distribution")
    ax.axhline(y=2, color="#dc2626", linestyle="--", alpha=0.7, label="Min for triplets (count≥2)")
    ax.legend()

    # Right: Histogram of frequencies
    ax = axes[1]
    ax.hist(freq.values, bins=50, color="#2563eb", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Number of Occurrences")
    ax.set_ylabel("Number of Misconceptions")
    ax.set_title("Distribution of Misconception Frequencies")
    ax.axvline(x=2, color="#dc2626", linestyle="--", alpha=0.7, label="Min for triplets")
    ax.legend()

    fig.suptitle("Long-Tail Distribution of Misconceptions in the Eedi Dataset", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "misconception_frequency_distribution.png", output_dir)

    # Compute summary statistics
    stats = {
        "total_unique": len(freq),
        "singletons": int((freq == 1).sum()),
        "doubletons": int((freq == 2).sum()),
        "count_gte_2": int((freq >= 2).sum()),
        "count_gte_5": int((freq >= 5).sum()),
        "count_gte_10": int((freq >= 10).sum()),
        "count_gte_20": int((freq >= 20).sum()),
        "max_freq": int(freq.max()),
        "mean_freq": float(freq.mean()),
        "median_freq": float(freq.median()),
    }
    print(f"    Singletons (count=1): {stats['singletons']} — CANNOT form triplets")
    print(f"    Viable for triplets (count≥2): {stats['count_gte_2']}")
    print(f"    Max frequency: {stats['max_freq']}, Median: {stats['median_freq']}")

    return stats


def plot_subject_distribution(train_df: pd.DataFrame, output_dir) -> dict:
    """
    Plot distribution of questions across mathematical subjects.

    Understanding subject imbalance helps design stratified evaluation
    and in-subject negative sampling.
    """
    subject_counts = train_df["SubjectName"].value_counts()

    fig, ax = plt.subplots(figsize=(12, 8))
    top_n = 25
    subject_counts.head(top_n).plot(
        kind="barh", ax=ax, color="#2563eb", edgecolor="white", alpha=0.8
    )
    ax.set_xlabel("Number of Questions")
    ax.set_ylabel("")
    ax.set_title(f"Top {top_n} Subjects by Question Count")
    ax.invert_yaxis()
    fig.tight_layout()
    save_figure(fig, "subject_distribution.png", output_dir)

    return {
        "total_subjects": len(subject_counts),
        "top_subject": subject_counts.index[0],
        "top_count": int(subject_counts.iloc[0]),
        "min_count": int(subject_counts.min()),
    }


def plot_construct_distribution(train_df: pd.DataFrame, output_dir) -> dict:
    """
    Plot distribution of questions across constructs.

    Constructs are more fine-grained than subjects. Many constructs
    have very few questions, which affects negative sampling.
    """
    construct_counts = train_df["ConstructName"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(construct_counts.values, bins=30, color="#16a34a", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Questions per Construct")
    ax.set_ylabel("Number of Constructs")
    ax.set_title(f"Distribution of Questions per Construct (N={len(construct_counts)} constructs)")
    fig.tight_layout()
    save_figure(fig, "construct_distribution.png", output_dir)

    return {
        "total_constructs": len(construct_counts),
        "mean_questions_per_construct": float(construct_counts.mean()),
        "median_questions_per_construct": float(construct_counts.median()),
        "single_question_constructs": int((construct_counts == 1).sum()),
    }


def analyze_misconception_construct_cooccurrence(qdp_df: pd.DataFrame, output_dir) -> dict:
    """
    Analyze how misconceptions span across constructs and subjects.

    If misconceptions are confined to a single construct, in-subject negatives
    are less meaningful. If they span many constructs, cross-construct
    retrieval becomes important.
    """
    # Number of unique constructs per misconception
    misc_construct = (
        qdp_df.groupby("MisconceptionId")["ConstructId"]
        .nunique()
        .reset_index()
        .rename(columns={"ConstructId": "n_constructs"})
    )
    misc_subject = (
        qdp_df.groupby("MisconceptionId")["SubjectId"]
        .nunique()
        .reset_index()
        .rename(columns={"SubjectId": "n_subjects"})
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(misc_construct["n_constructs"].values, bins=range(1, misc_construct["n_constructs"].max() + 2),
            color="#f59e0b", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Number of Unique Constructs")
    ax.set_ylabel("Number of Misconceptions")
    ax.set_title("Constructs per Misconception")

    ax = axes[1]
    ax.hist(misc_subject["n_subjects"].values, bins=range(1, misc_subject["n_subjects"].max() + 2),
            color="#ef4444", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Number of Unique Subjects")
    ax.set_ylabel("Number of Misconceptions")
    ax.set_title("Subjects per Misconception")

    fig.suptitle("Misconception Span Across Constructs and Subjects", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "misconception_cooccurrence.png", output_dir)

    stats = {
        "single_construct_misconceptions": int((misc_construct["n_constructs"] == 1).sum()),
        "multi_construct_misconceptions": int((misc_construct["n_constructs"] > 1).sum()),
        "mean_constructs_per_misconception": float(misc_construct["n_constructs"].mean()),
        "max_constructs_per_misconception": int(misc_construct["n_constructs"].max()),
        "single_subject_misconceptions": int((misc_subject["n_subjects"] == 1).sum()),
        "multi_subject_misconceptions": int((misc_subject["n_subjects"] > 1).sum()),
    }
    print(f"    Misconceptions in exactly 1 construct: {stats['single_construct_misconceptions']}")
    print(f"    Misconceptions spanning >1 construct:   {stats['multi_construct_misconceptions']}")

    return stats


def analyze_question_text(train_df: pd.DataFrame, output_dir) -> dict:
    """
    Analyze question text properties: length, LaTeX prevalence, image references.

    LaTeX content may confuse the SentenceTransformer tokenizer.
    Very long questions may be truncated during encoding.
    """
    texts = train_df["QuestionText"]
    lengths = texts.str.len()
    has_latex = texts.str.contains(r"\\[(\[]|\\frac|\\mathrm|\\times", regex=True)
    has_image = texts.str.contains(r"!\[", regex=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(lengths.values, bins=50, color="#8b5cf6", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Character Length")
    ax.set_ylabel("Number of Questions")
    ax.set_title("Question Text Length Distribution")
    ax.axvline(x=512, color="#dc2626", linestyle="--", alpha=0.7, label="Typical token limit")
    ax.legend()

    ax = axes[1]
    categories = ["Has LaTeX", "Has Images", "Plain Text"]
    counts = [
        int(has_latex.sum()),
        int(has_image.sum()),
        int((~has_latex & ~has_image).sum()),
    ]
    colors = ["#2563eb", "#f59e0b", "#16a34a"]
    ax.bar(categories, counts, color=colors, edgecolor="white", alpha=0.8)
    ax.set_ylabel("Number of Questions")
    ax.set_title("Question Content Types")
    for i, (c, v) in enumerate(zip(categories, counts)):
        ax.text(i, v + 10, str(v), ha="center", fontsize=10)

    fig.tight_layout()
    save_figure(fig, "question_text_analysis.png", output_dir)

    stats = {
        "mean_length": float(lengths.mean()),
        "median_length": float(lengths.median()),
        "max_length": int(lengths.max()),
        "pct_latex": float(has_latex.mean() * 100),
        "pct_images": float(has_image.mean() * 100),
        "pct_plain": float((~has_latex & ~has_image).mean() * 100),
    }
    print(f"    LaTeX questions: {stats['pct_latex']:.1f}%")
    print(f"    Image questions: {stats['pct_images']:.1f}%")
    print(f"    Mean text length: {stats['mean_length']:.0f} chars")

    return stats


def analyze_label_coverage(train_df: pd.DataFrame, output_dir) -> dict:
    """
    Analyze what fraction of distractors have misconception labels.

    Missing labels reduce the training data for triplet construction.
    """
    # Total possible distractors: 3 per question
    total_distractors = len(train_df) * 3

    # Count labeled distractors
    labeled = 0
    for _, row in train_df.iterrows():
        correct = row["CorrectAnswer"]
        for letter in ["A", "B", "C", "D"]:
            if letter == correct:
                continue
            col = f"Misconception{letter}Id"
            if not pd.isna(row[col]):
                labeled += 1

    unlabeled = total_distractors - labeled
    pct_labeled = labeled / total_distractors * 100

    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = [labeled, unlabeled]
    labels = [f"Labeled\n({labeled:,}, {pct_labeled:.1f}%)",
              f"Unlabeled\n({unlabeled:,}, {100-pct_labeled:.1f}%)"]
    colors = ["#2563eb", "#d1d5db"]
    wedges, texts = ax.pie(
        sizes, labels=labels, colors=colors,
        startangle=90, textprops={"fontsize": 12}
    )
    ax.set_title("Misconception Label Coverage\n(Distractors with Known Misconceptions)")
    fig.tight_layout()
    save_figure(fig, "label_coverage.png", output_dir)

    stats = {
        "total_distractors": total_distractors,
        "labeled": labeled,
        "unlabeled": unlabeled,
        "pct_labeled": pct_labeled,
    }
    print(f"    Labeled distractors: {labeled:,} / {total_distractors:,} ({pct_labeled:.1f}%)")

    return stats


def analyze_qdp_per_question(qdp_df: pd.DataFrame, output_dir) -> dict:
    """
    Analyze how many labeled QDPs each question contributes.

    Questions with more labeled distractors contribute more training signal.
    """
    qdp_per_q = qdp_df.groupby("QuestionId").size()

    fig, ax = plt.subplots(figsize=(8, 5))
    qdp_per_q.value_counts().sort_index().plot(
        kind="bar", ax=ax, color="#2563eb", edgecolor="white", alpha=0.8
    )
    ax.set_xlabel("Number of Labeled Distractors per Question")
    ax.set_ylabel("Number of Questions")
    ax.set_title("Distribution of Labeled Distractors per Question")
    for i, (idx, val) in enumerate(qdp_per_q.value_counts().sort_index().items()):
        ax.text(i, val + 5, str(val), ha="center", fontsize=10)
    fig.tight_layout()
    save_figure(fig, "qdp_per_question.png", output_dir)

    return {
        "mean_qdp_per_q": float(qdp_per_q.mean()),
        "questions_with_1": int((qdp_per_q == 1).sum()),
        "questions_with_2": int((qdp_per_q == 2).sum()),
        "questions_with_3": int((qdp_per_q == 3).sum()),
    }
