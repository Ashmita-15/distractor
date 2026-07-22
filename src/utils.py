"""
Shared utilities for the Misconception-Aware Distractor Generation project.

Provides helper functions for reproducibility, logging, and I/O.
"""

import random
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from src.config import RANDOM_SEED, FIGURE_STYLE


def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds across all libraries for reproducibility.

    Args:
        seed: Integer seed value. Default from config.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_matplotlib() -> None:
    """Apply publication-quality matplotlib settings from config."""
    plt.rcParams.update(FIGURE_STYLE)


def save_figure(fig: plt.Figure, filename: str, output_dir: Path) -> Path:
    """
    Save a matplotlib figure to disk.

    Args:
        fig: Matplotlib figure object.
        filename: Name of the output file (e.g., 'misconception_dist.png').
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [Saved] {filepath}")
    return filepath


def save_results(results: dict, filename: str, output_dir: Path) -> Path:
    """
    Save evaluation results as JSON.

    Args:
        results: Dictionary of results.
        filename: Name of the output file.
        output_dir: Directory to save results.

    Returns:
        Path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"  [Saved] {filepath}")
    return filepath


def get_device() -> torch.device:
    """
    Get the best available compute device.

    Returns:
        torch.device for CUDA or CPU.
        Note: MPS is intentionally excluded because PyTorch 2.2 + MPS
        is extremely slow for small SentenceTransformer models.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_section_header(title: str) -> None:
    """Print a formatted section header for console output."""
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width + "\n")
