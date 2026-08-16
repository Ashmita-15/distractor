"""
Central configuration for the Misconception-Aware Distractor Generation project.

All paths, hyperparameters, and constants are defined here.
This avoids hardcoding across modules and enables reproducibility.
"""

import os
from pathlib import Path


# =============================================================================
# Project Paths
# =============================================================================

# Root of the project (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_path(var_name: str, default: Path) -> Path:
    """Resolve a path from an environment variable, falling back to default.

    Enables cloud environments (Colab, Kaggle) to redirect data/output
    locations without editing code, e.g. Kaggle mounts competition data
    read-only under /kaggle/input/.
    """
    value = os.environ.get(var_name)
    return Path(value) if value else default


# Data paths (override with DISTRACTOR_DATA_DIR)
DATA_DIR = _env_path("DISTRACTOR_DATA_DIR", PROJECT_ROOT / "datasets")
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
MISCONCEPTION_CSV = DATA_DIR / "misconception_mapping.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

# Output paths (override with DISTRACTOR_OUTPUT_DIR)
OUTPUT_DIR = _env_path("DISTRACTOR_OUTPUT_DIR", PROJECT_ROOT / "outputs")
FIGURES_DIR = OUTPUT_DIR / "figures"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
TRIPLETS_DIR = OUTPUT_DIR / "triplets"

# Create output directories
for d in [FIGURES_DIR, EMBEDDINGS_DIR, MODELS_DIR, RESULTS_DIR, TRIPLETS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Random Seed (Reproducibility)
# =============================================================================

RANDOM_SEED = 42


# =============================================================================
# Model Configuration
# =============================================================================

# Pretrained SentenceTransformer for baseline
BASELINE_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Output dimension of all-MiniLM-L6-v2

# Fine-tuning hyperparameters
# Epochs: 3 (not more) — the triplet set is small (~8K static triplets), so
# additional epochs overfit rather than help; the best checkpoint is selected
# by validation TripletEvaluator accuracy. Also keeps CPU wall-time feasible
# (~1h/epoch on this machine; MPS benchmarked 5x slower than CPU for MiniLM
# on torch 2.2, so CPU is used).
FINETUNE_EPOCHS = 3
FINETUNE_BATCH_SIZE = 32
FINETUNE_LR = 2e-5
FINETUNE_WARMUP_RATIO = 0.1
TRIPLET_MARGIN = 0.5
EVAL_STEPS = 100


# =============================================================================
# Retrieval Configuration
# =============================================================================

# Top-K values for evaluation
TOP_K_VALUES = [1, 3, 5, 10, 25]

# Maximum K for FAISS retrieval (retrieve more, evaluate at different K)
MAX_K = 50


# =============================================================================
# Data Split Configuration
# =============================================================================

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


# =============================================================================
# Text Representation Template
# =============================================================================

# Template for constructing the text representation of a question-distractor pair.
# This is the canonical representation used for training and for the corpus in
# every experiment; it is unchanged.
QDP_TEXT_TEMPLATE = (
    "Subject: {subject} | Construct: {construct} | "
    "Question: {question} | Correct Answer: {correct_answer} | "
    "Distractor: {distractor}"
)

# Named query-representation variants for the query-representation ablation.
# Only the QUERY side varies; the corpus keeps QDP_TEXT_TEMPLATE ("full"),
# mirroring deployment, where historical corpus items have known distractors
# but an incoming question does not.
#   full            - current system (reference condition)
#   no_distractor   - metadata retained, distractor removed
#   question_answer - deployment-faithful: no metadata, no distractor
QDP_TEXT_TEMPLATES = {
    "full": QDP_TEXT_TEMPLATE,
    "no_distractor": (
        "Subject: {subject} | Construct: {construct} | "
        "Question: {question} | Correct Answer: {correct_answer}"
    ),
    "question_answer": (
        "Question: {question} | Correct Answer: {correct_answer}"
    ),
}


# =============================================================================
# Visualization Configuration
# =============================================================================

# Matplotlib style settings for publication-quality figures
FIGURE_DPI = 150
FIGURE_STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (10, 6),
    "figure.dpi": FIGURE_DPI,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
}
