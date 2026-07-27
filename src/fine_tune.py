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
import json
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
from sentence_transformers.training_args import BatchSamplers


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
    save_steps: Optional[int] = None,
    save_total_limit: Optional[int] = 2,
    output_dir: Optional[Path] = None,
    seed: int = RANDOM_SEED,
    objective: str = "triplet",
) -> Tuple[SentenceTransformer, Path]:
    """
    Fine-tune a SentenceTransformer with the selected training objective.

    objective (the ONLY thing that differs between experiments):
        "triplet" (default) — TripletLoss over (anchor, positive, negative)
            triplets with length-grouped batching. Reproduces Experiments 1-2
            byte-for-byte.
        "mnrl" (Experiment 3) — MultipleNegativesRankingLoss over
            (anchor, positive) pairs (negatives drawn in-batch), with the
            NO_DUPLICATES batch sampler (the officially recommended MNRL
            setup). Everything else — data source, seed, LR, optimizer,
            scheduler, epochs, checkpoint cadence, evaluator — is unchanged.

    In-batch false negatives (mnrl): left unmasked, faithful to the standard
    MNRL formulation, and documented as a limitation in the experiment report.

    Checkpoint cadence (Experiment 2 support):
        save_steps defaults to eval_steps and save_total_limit to 2, which
        reproduces the Experiment 1 configuration exactly. For trajectory
        studies pass save_steps=50, eval_steps=50, save_total_limit=None
        (keep all checkpoints). Note the HF Trainer requires save_steps to
        be a round multiple of eval_steps when load_best_model_at_end=True.
        Saving/evaluating more often does not alter the training trajectory:
        evaluation runs in eval mode under no_grad and the data order comes
        from an isolated torch.Generator.

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
        save_steps: Save a checkpoint every N steps (default: eval_steps).
        save_total_limit: Max checkpoints kept (None = keep all).
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

    if objective not in ("triplet", "mnrl"):
        raise ValueError(f"objective must be 'triplet' or 'mnrl', got {objective!r}")

    lengths = None  # only used by the triplet path (length-grouped batching)
    if objective == "mnrl":
        # (anchor, positive) pairs — negatives are constructed in-batch by
        # MultipleNegativesRankingLoss. Keeping every row (no dedup) preserves
        # the triplet run's step budget so checkpoints land on the same steps.
        from src.triplet_construction import triplets_to_pairs
        pairs = triplets_to_pairs(train_triplets)
        train_dataset = Dataset.from_dict({
            "anchor": [a for a, _ in pairs],
            "positive": [p for _, p in pairs],
        })
        print(f"  Training examples: {len(train_dataset):,} (anchor, positive) pairs")
        # scale left at the official default (20.0)
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
    else:
        # Triplet columns must be ordered (anchor, positive, negative) — the
        # loss consumes dataset columns positionally.
        train_dataset = Dataset.from_dict({
            "anchor": [t[0] for t in train_triplets],
            "positive": [t[1] for t in train_triplets],
            "negative": [t[2] for t in train_triplets],
        })
        print(f"  Training examples: {len(train_dataset):,}")

        # Length keys for length-grouped batching: the collator pads each
        # column to its own batch max, so the max token count across the three
        # texts is the right per-triplet grouping key. Token counts are cached
        # per unique text (anchors repeat across triplets).
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

    if save_steps is None:
        save_steps = eval_steps

    total_steps = math.ceil(len(train_dataset) / batch_size) * epochs

    # MNRL draws negatives in-batch, so batch composition matters: NO_DUPLICATES
    # is the officially recommended sampler (avoids a pair's duplicate landing
    # in the same batch as a self-false-negative). The triplet path ignores
    # this — its LengthGroupedTripletTrainer overrides the sampler — so this
    # setting is inert there and does not change Experiments 1-2.
    batch_sampler = (
        BatchSamplers.NO_DUPLICATES if objective == "mnrl"
        else BatchSamplers.BATCH_SAMPLER
    )

    print(f"\n  === Training Configuration ===")
    print(f"  Objective:     {objective}")
    print(f"  Epochs:        {epochs}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Warmup ratio:  {warmup_ratio}")
    print(f"  Total steps:   ~{total_steps}" + (" (NO_DUPLICATES may drop a few)" if objective == "mnrl" else ""))
    if objective == "triplet":
        print(f"  Margin:        {margin}")
    print(f"  Batch sampler: {batch_sampler}")
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
        batch_sampler=batch_sampler,
        eval_strategy="steps" if evaluator else "no",
        eval_steps=eval_steps,
        save_strategy="steps" if evaluator else "no",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=bool(evaluator),
        metric_for_best_model="eval_val_cosine_accuracy",
        greater_is_better=True,
        use_cpu=use_cpu,
        seed=seed,
        logging_steps=50,
        report_to="none",
    )

    if objective == "mnrl":
        # Standard trainer; batch sampling is governed by args.batch_sampler
        # (NO_DUPLICATES). No length grouping — batch composition is the
        # negative sampling and must follow the recommended MNRL setup.
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            loss=train_loss,
            evaluator=evaluator,
        )
    else:
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

    # Persist the training log (loss curve, eval accuracy per step) so the
    # run is auditable after the fact — required for reporting.
    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"  [Saved] {log_path}")

    print(f"\n  ✅ Training complete. Model saved to: {output_dir}")

    return model, output_dir


def run_training(
    train_triplets_file: str = "train_triplets.jsonl",
    val_triplets_file: str = "val_triplets.jsonl",
    output_dir: Optional[Path] = None,
    epochs: int = FINETUNE_EPOCHS,
    batch_size: int = FINETUNE_BATCH_SIZE,
    eval_steps: int = EVAL_STEPS,
    save_steps: Optional[int] = None,
    save_total_limit: Optional[int] = 2,
    max_train_triplets: Optional[int] = None,
    objective: str = "triplet",
    seed: int = RANDOM_SEED,
) -> Path:
    """
    Self-contained training entry point: load prepared triplets, fine-tune,
    save the best checkpoint and training log. Performs NO preprocessing,
    NO baseline retrieval, and NO retrieval evaluation — those live in
    their own pipeline stages.

    Args:
        train_triplets_file: JSONL file in TRIPLETS_DIR (from 03_create_triplets).
        val_triplets_file: JSONL validation triplets for best-checkpoint selection.
        output_dir: Model output directory (default MODELS_DIR/finetuned_pedagogical).
        epochs: Training epochs.
        batch_size: Per-device batch size.
        max_train_triplets: Optional cap on training triplets (smoke tests only).
        objective: "triplet" (Exp 1-2) or "mnrl" (Exp 3). See finetune_model.
            The same train_triplets file is used either way; for "mnrl" the
            negatives are discarded to form (anchor, positive) pairs.
        seed: Random seed. Controls training stochasticity only (batch order,
            dropout); the encoder init, triplet file, and splits are fixed, so
            varying this isolates run-to-run variance for replication studies.

    Returns:
        Path to the saved model directory.
    """
    from src.triplet_construction import load_triplets

    train_triplets = load_triplets(train_triplets_file)
    val_triplets = load_triplets(val_triplets_file)

    if max_train_triplets is not None:
        train_triplets = train_triplets[:max_train_triplets]
        print(f"  ⚠ Smoke test: capped to {len(train_triplets)} training triplets")

    if output_dir is None:
        output_dir = MODELS_DIR / "finetuned_pedagogical"

    _, model_path = finetune_model(
        train_triplets=train_triplets,
        val_triplets=val_triplets,
        epochs=epochs,
        batch_size=batch_size,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        output_dir=Path(output_dir),
        objective=objective,
        seed=seed,
    )
    return model_path
