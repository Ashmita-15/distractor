"""
Fine-tuning SentenceTransformer with misconception-aware triplet loss.

This module fine-tunes the same all-MiniLM-L6-v2 model used as the baseline,
but with triplet loss supervision from misconception labels. This ensures
a fair comparison: both models have the same architecture, with the only
difference being the training signal.

Training Pipeline:
    1. Convert triplets to InputExamples
    2. Create DataLoader
    3. Configure TripletLoss with distance metric
    4. Train with early stopping based on validation loss
    5. Save best checkpoint
"""

import os
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers.trainer_pt_utils import LengthGroupedSampler
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    InputExample,
    losses,
    evaluation,
)
from sentence_transformers.sampler import DefaultBatchSampler


class LengthGroupedTripletTrainer(SentenceTransformerTrainer):
    """
    SentenceTransformerTrainer with length-grouped batching.

    Why: training cost on CPU is dominated by padded sequence length. Our
    texts have median ~94 tokens but p99 ~248, so a randomly sampled batch
    of 96 texts pads to ~246 tokens — over half the transformer compute is
    spent on padding. Grouping batches by length (megabatch shuffle + sort,
    via transformers' LengthGroupedSampler) makes batches length-homogeneous,
    cutting wall time ~2x with NO truncation and NO loss of training data.
    Randomness is preserved at the megabatch level, and batch order is
    re-shuffled each epoch.

    Args:
        lengths: Per-example length keys (max token count across the
            anchor/positive/negative columns), aligned with the train dataset.
    """

    def __init__(self, *args, lengths: Optional[List[int]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._lengths = lengths

    def get_batch_sampler(
        self,
        dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns=None,
        generator=None,
    ):
        if self._lengths is None:
            return super().get_batch_sampler(
                dataset, batch_size, drop_last,
                valid_label_columns=valid_label_columns, generator=generator,
            )
        return DefaultBatchSampler(
            LengthGroupedSampler(
                batch_size=batch_size,
                lengths=self._lengths,
                generator=generator,
            ),
            batch_size=batch_size,
            drop_last=drop_last,
        )

from src.config import (
    BASELINE_MODEL_NAME,
    FINETUNE_EPOCHS,
    FINETUNE_BATCH_SIZE,
    FINETUNE_LR,
    FINETUNE_WARMUP_RATIO,
    TRIPLET_MARGIN,
    EVAL_STEPS,
    MODELS_DIR,
    RANDOM_SEED,
)
from src.utils import set_seed, get_device


def create_train_dataloader(
    triplets: List[Tuple[str, str, str]],
    batch_size: int = FINETUNE_BATCH_SIZE,
) -> DataLoader:
    """
    Create a DataLoader from triplets for SentenceTransformer training.

    Args:
        triplets: List of (anchor, positive, negative) text tuples.
        batch_size: Training batch size.

    Returns:
        DataLoader of InputExample objects.
    """
    examples = [
        InputExample(texts=[a, p, n])
        for a, p, n in triplets
    ]
    print(f"  Training examples: {len(examples):,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Steps per epoch: {math.ceil(len(examples) / batch_size)}")

    dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    return dataloader


def create_evaluator(
    val_triplets: List[Tuple[str, str, str]],
    name: str = "val",
) -> evaluation.TripletEvaluator:
    """
    Create a TripletEvaluator for validation during training.

    The evaluator computes accuracy: the fraction of triplets where
    dist(anchor, positive) < dist(anchor, negative).

    Args:
        val_triplets: Validation triplets.
        name: Name for logging.

    Returns:
        TripletEvaluator instance.
    """
    anchors = [t[0] for t in val_triplets]
    positives = [t[1] for t in val_triplets]
    negatives = [t[2] for t in val_triplets]

    evaluator = evaluation.TripletEvaluator(
        anchors=anchors,
        positives=positives,
        negatives=negatives,
        name=name,
        main_distance_function="cosine",
    )
    print(f"  Validation triplets: {len(val_triplets):,}")
    return evaluator


def finetune_model(
    train_triplets: List[Tuple[str, str, str]],
    val_triplets: Optional[List[Tuple[str, str, str]]] = None,
    model_name: str = BASELINE_MODEL_NAME,
    epochs: int = FINETUNE_EPOCHS,
    batch_size: int = FINETUNE_BATCH_SIZE,
    learning_rate: float = FINETUNE_LR,
    warmup_ratio: float = FINETUNE_WARMUP_RATIO,
    margin: float = TRIPLET_MARGIN,
    eval_steps: int = EVAL_STEPS,
    output_dir: Optional[Path] = None,
    seed: int = RANDOM_SEED,
) -> Tuple[SentenceTransformer, Path]:
    """
    Fine-tune a SentenceTransformer using triplet loss.

    Args:
        train_triplets: Training triplets.
        val_triplets: Validation triplets (optional, for early stopping).
        model_name: Base model to fine-tune.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate for AdamW.
        warmup_ratio: Fraction of steps for linear warmup.
        margin: Triplet loss margin.
        eval_steps: Evaluate every N steps.
        output_dir: Directory to save the fine-tuned model.
        seed: Random seed.

    Returns:
        Tuple of (fine-tuned model, save path).
    """
    set_seed(seed)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = MODELS_DIR / f"finetuned_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base model.
    # Device note: get_device() returns CPU on this machine (MPS benchmarked
    # 5x slower than CPU for MiniLM on torch 2.2, and MPS OOMs at batch 32).
    # The HF Trainer auto-detects MPS regardless of the model's device, so
    # use_cpu must be passed explicitly in the training arguments below.
    device = get_device()
    use_cpu = device.type == "cpu"
    if use_cpu:
        # args.use_cpu alone is NOT enough: the HF Trainer's internal
        # Accelerator selects its own device (MPS here) independently of the
        # training arguments, moving input batches to MPS while the model
        # stays on CPU. ACCELERATE_USE_CPU is read at AcceleratorState init
        # and forces the whole stack onto CPU. (Must be set before the
        # Trainer is constructed.)
        os.environ["ACCELERATE_USE_CPU"] = "true"
    print(f"\n  Loading base model: {model_name}")
    print(f"  Device: {device}")
    model = SentenceTransformer(model_name, device=str(device))

    # Triplet columns must be ordered (anchor, positive, negative) — the
    # loss consumes dataset columns positionally.
    train_dataset = Dataset.from_dict({
        "anchor": [t[0] for t in train_triplets],
        "positive": [t[1] for t in train_triplets],
        "negative": [t[2] for t in train_triplets],
    })
    print(f"  Training examples: {len(train_dataset):,}")

    # Length keys for length-grouped batching: the collator pads each column
    # to its own batch max, so the max token count across the three texts is
    # the right per-triplet grouping key. Token counts are cached per unique
    # text (anchors repeat across triplets).
    tokenizer = model.tokenizer
    unique_texts = list({t for trip in train_triplets for t in trip})
    token_counts = [len(ids) for ids in tokenizer(unique_texts)["input_ids"]]
    token_len = dict(zip(unique_texts, token_counts))
    lengths = [max(token_len[a], token_len[p], token_len[n]) for a, p, n in train_triplets]
    print(f"  Length-grouped batching over {len(unique_texts):,} unique texts "
          f"(median length key: {int(np.median(lengths))} tokens)")

    # Configure TripletLoss
    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=margin,
    )

    # Create evaluator
    evaluator = None
    if val_triplets and len(val_triplets) > 0:
        evaluator = create_evaluator(val_triplets)

    total_steps = math.ceil(len(train_dataset) / batch_size) * epochs

    print(f"\n  === Training Configuration ===")
    print(f"  Epochs:        {epochs}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Warmup ratio:  {warmup_ratio}")
    print(f"  Total steps:   {total_steps}")
    print(f"  Margin:        {margin}")
    print(f"  Eval steps:    {eval_steps}")
    print(f"  Output:        {output_dir}")
    print(f"\n  Starting training...\n")

    checkpoint_dir = output_dir / "checkpoints"
    args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        eval_strategy="steps" if evaluator else "no",
        eval_steps=eval_steps,
        save_strategy="steps" if evaluator else "no",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=bool(evaluator),
        metric_for_best_model="eval_val_cosine_accuracy",
        greater_is_better=True,
        use_cpu=use_cpu,
        seed=seed,
        logging_steps=50,
        report_to="none",
    )

    trainer = LengthGroupedTripletTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=evaluator,
        lengths=lengths,
    )
    trainer.train()

    # Save the best model (loaded at end when an evaluator is present)
    model.save(str(output_dir))

    print(f"\n  ✅ Training complete. Model saved to: {output_dir}")

    return model, output_dir
