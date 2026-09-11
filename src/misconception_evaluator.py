"""
Independent misconception-alignment evaluator.

Measures the project's core construct directly: does a distractor encode the
misconception the question's gold distractors encode?

Architecture — bi-encoder ranker over misconception descriptions.
    query    : Question + Correct Answer + Incorrect answer chosen
    document : misconception description text (all 2,587 in the taxonomy)
    scoring  : cosine similarity; the gold misconception's rank is the measure.

Why a ranker and not an N-way classifier: 27.6% of test gold misconceptions
never occur in the training split. A softmax over training classes assigns
those zero probability by construction, which would score those questions as
failures regardless of distractor quality and bias every comparison. Ranking
misconception *descriptions* keeps unseen labels rankable.

Independence from Stage 1 (mandatory, see check_independence):
    - different base encoder (mpnet vs the MiniLM used by the retriever)
    - different task: QDP -> misconception description, not QDP -> QDP
    - different target space and training pairs
    - trained only on the train split; never sees test questions or any
      generated text.

The evaluator input never contains the misconception name or the generator's
self-reported misconception; build_query_text is the single construction point
and assert_no_label_leakage enforces it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Query template. Deliberately excludes misconception text of any kind.
QUERY_TEMPLATE = ("Question: {question} | Correct answer: {correct_answer} "
                  "| Incorrect answer chosen: {distractor}")

RANK_KS = (1, 5, 10, 25)


# =============================================================================
# Query / training-pair construction
# =============================================================================

def build_query_text(question: str, correct_answer: str, distractor: str) -> str:
    """Single construction point for evaluator input."""
    return QUERY_TEMPLATE.format(question=question, correct_answer=correct_answer,
                                 distractor=distractor)


def build_queries_from_qdp(qdp: pd.DataFrame) -> List[str]:
    return [build_query_text(r["QuestionText"], r["CorrectAnswerText"],
                             r["DistractorText"]) for _, r in qdp.iterrows()]


def build_training_pairs(qdp: pd.DataFrame,
                         misconception_map: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    (query, misconception_description) pairs for MNRL.

    The description is the *target*, never part of the query, so the model
    learns to associate a wrong answer with the error that produces it rather
    than to copy a label.
    """
    name = dict(zip(misconception_map.MisconceptionId, misconception_map.MisconceptionName))
    pairs = []
    for _, r in qdp.iterrows():
        mid = r["MisconceptionId"]
        if pd.isna(mid) or int(mid) not in name:
            continue
        pairs.append((build_query_text(r["QuestionText"], r["CorrectAnswerText"],
                                       r["DistractorText"]), name[int(mid)]))
    return pairs


# =============================================================================
# Contamination checks (mandatory)
# =============================================================================

def assert_no_label_leakage(queries: Sequence[str],
                            misconception_map: pd.DataFrame) -> Dict:
    """Verify no misconception description appears verbatim in any query."""
    names = [n for n in misconception_map.MisconceptionName.astype(str) if len(n) > 15]
    hits = sum(any(n in q for n in names) for q in queries)
    out = {"queries_checked": len(queries), "queries_containing_a_label": hits}
    assert hits == 0, f"label leakage: {hits} queries contain a misconception name"
    return out


def assert_no_question_overlap(train_qdp: pd.DataFrame,
                               applied_qdp: pd.DataFrame) -> Dict:
    """Verify evaluator training questions are disjoint from application questions."""
    overlap = set(train_qdp.QuestionId) & set(applied_qdp.QuestionId)
    out = {"train_questions": int(train_qdp.QuestionId.nunique()),
           "applied_questions": int(applied_qdp.QuestionId.nunique()),
           "overlap": len(overlap)}
    assert not overlap, f"{len(overlap)} questions appear in both training and application"
    return out


def check_independence(evaluator_base: str, stage1_base: str = "all-MiniLM-L6-v2") -> Dict:
    """Verify the evaluator is not built on the Stage 1 retriever."""
    distinct = stage1_base.lower() not in str(evaluator_base).lower()
    out = {"evaluator_base": evaluator_base, "stage1_base": stage1_base,
           "distinct_base_encoder": distinct}
    assert distinct, ("evaluator must not be built on the Stage 1 retriever's "
                      "base encoder")
    return out


# =============================================================================
# Ranker
# =============================================================================

class MisconceptionRanker:
    """
    Ranks the full misconception taxonomy for a given (question, answer,
    distractor) query. Label embeddings are computed once and reused.
    """

    def __init__(self, model_path: str, misconception_map: pd.DataFrame,
                 batch_size: int = 128):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(str(model_path))
        self.ids = misconception_map.MisconceptionId.values.astype(int)
        self.id_to_col = {int(m): i for i, m in enumerate(self.ids)}
        self.label_emb = self.model.encode(
            misconception_map.MisconceptionName.astype(str).tolist(),
            normalize_embeddings=True, show_progress_bar=False, batch_size=256)
        self.batch_size = batch_size

    def rank_positions(self, queries: Sequence[str]) -> np.ndarray:
        """
        Returns (n_queries, n_labels): the 1-based rank of each label column.
        """
        qe = self.model.encode(list(queries), normalize_embeddings=True,
                               show_progress_bar=False, batch_size=self.batch_size)
        sims = qe @ self.label_emb.T
        order = np.argsort(-sims, axis=1)
        pos = np.empty_like(order)
        np.put_along_axis(pos, order, np.arange(order.shape[1]), axis=1)
        return pos + 1

    def best_gold_ranks(self, queries: Sequence[str],
                        gold_id_sets: Sequence[Iterable[int]]) -> np.ndarray:
        """
        Best (lowest) rank achieved by any of the question's gold misconceptions.

        A generated distractor has no designated gold label, so alignment means
        matching *any* misconception the question's teacher-written distractors
        encode.
        """
        pos = self.rank_positions(queries)
        out = np.full(len(queries), np.iinfo(np.int32).max, dtype=np.int64)
        for i, golds in enumerate(gold_id_sets):
            cols = [self.id_to_col[int(g)] for g in golds if int(g) in self.id_to_col]
            if cols:
                out[i] = int(pos[i, cols].min())
        return out


# =============================================================================
# Metrics
# =============================================================================

def rank_metrics(ranks: np.ndarray, ks: Sequence[int] = RANK_KS) -> Dict[str, float]:
    ranks = np.asarray(ranks, dtype=float)
    valid = np.isfinite(ranks) & (ranks < np.iinfo(np.int32).max)
    if valid.sum() == 0:
        return {"n": 0, "MRR": float("nan"), **{f"Recall@{k}": float("nan") for k in ks}}
    r = ranks[valid]
    out = {"n": int(valid.sum()), "MRR": float((1.0 / r).mean())}
    for k in ks:
        out[f"Recall@{k}"] = float((r <= k).mean())
    return out


def reciprocal_ranks(ranks: np.ndarray) -> np.ndarray:
    r = np.asarray(ranks, dtype=float)
    rr = 1.0 / r
    rr[~np.isfinite(rr)] = 0.0
    return rr


# =============================================================================
# Training
# =============================================================================

def train_evaluator(
    train_pairs: List[Tuple[str, str]],
    dev_pairs: Optional[List[Tuple[str, str]]],
    output_dir: Path,
    base_model: str = "sentence-transformers/all-mpnet-base-v2",
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    seed: int = 42,
) -> Path:
    """
    Fine-tune the ranker with MNRL on (query, misconception description) pairs.

    NO_DUPLICATES batching matters here: many QDPs share a misconception, so an
    unconstrained batch would place identical descriptions in the same batch as
    in-batch negatives of each other. The seeded trainer is reused from the
    Stage 1 module (imported, not modified) because it propagates the run seed
    into that sampler.
    """
    import os
    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, losses,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.training_args import BatchSamplers
    from src.fine_tune import SeededNoDuplicatesTrainer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_cpu = not torch.cuda.is_available()
    if use_cpu:
        os.environ["ACCELERATE_USE_CPU"] = "true"

    model = SentenceTransformer(base_model)
    ds = Dataset.from_dict({"anchor": [a for a, _ in train_pairs],
                            "positive": [p for _, p in train_pairs]})
    loss = losses.MultipleNegativesRankingLoss(model=model)

    evaluator = None
    if dev_pairs:
        from sentence_transformers.evaluation import InformationRetrievalEvaluator
        queries = {str(i): a for i, (a, _) in enumerate(dev_pairs)}
        corpus_texts = sorted({p for _, p in dev_pairs})
        corpus = {str(j): t for j, t in enumerate(corpus_texts)}
        text_to_cid = {t: str(j) for j, t in enumerate(corpus_texts)}
        relevant = {str(i): {text_to_cid[p]} for i, (_, p) in enumerate(dev_pairs)}
        evaluator = InformationRetrievalEvaluator(queries, corpus, relevant,
                                                  name="dev", show_progress_bar=False)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch" if evaluator else "no",
        save_strategy="epoch" if evaluator else "no",
        save_total_limit=1,
        load_best_model_at_end=bool(evaluator),
        use_cpu=use_cpu,
        seed=seed,
        logging_steps=50,
        report_to="none",
    )
    trainer = SeededNoDuplicatesTrainer(model=model, args=args, train_dataset=ds,
                                        loss=loss, evaluator=evaluator)
    trainer.train()
    model.save(str(output_dir / "model"))
    return output_dir / "model"


# =============================================================================
# Positive control (mandatory before drawing conclusions)
# =============================================================================

_NUM = __import__("re").compile(r"-?\d+\.?\d*")


def perturb_numeric(s: str) -> str:
    """Digit-perturb a numeric answer; the near-miss control."""
    m = _NUM.search(str(s))
    if not m:
        return str(s) + " (altered)"
    try:
        v = float(m.group())
    except ValueError:
        return str(s) + " (altered)"
    new = v + (1 if abs(v) < 100 else 10)
    new_s = str(int(new)) if float(new).is_integer() else f"{new:g}"
    return str(s)[:m.start()] + new_s + str(s)[m.end():]


def positive_control(ranker: MisconceptionRanker, qdp: pd.DataFrame,
                     corpus_distractors: Sequence[str], seed: int = 42,
                     limit: Optional[int] = None) -> pd.DataFrame:
    """
    Score conditions of known quality, then measure discrimination via AUC.

    C1 gold vs random distractor     - coarse; the evaluator must pass this
    C2 gold vs the correct answer    - coarse; must pass
    C3 gold vs digit-perturbed gold  - near-miss; reported as a probe. Every
       instrument tested in this project so far scores ~0.56 here.
    """
    from scipy import stats as st

    rng = np.random.RandomState(seed)
    df = qdp if limit is None else qdp.head(limit)
    rows = []
    for _, r in df.iterrows():
        gold_d = str(r["DistractorText"])
        variants = [
            ("gold", gold_d),
            ("random", str(corpus_distractors[rng.randint(len(corpus_distractors))])),
            ("correct_answer", str(r["CorrectAnswerText"])),
            ("perturbed_gold", perturb_numeric(gold_d)),
        ]
        for kind, d in variants:
            rows.append({"kind": kind, "QuestionId": r["QuestionId"],
                         "gold_id": int(r["MisconceptionId"]),
                         "query": build_query_text(r["QuestionText"],
                                                   r["CorrectAnswerText"], d)})
    probe = pd.DataFrame(rows)
    ranks = ranker.best_gold_ranks(probe["query"].tolist(),
                                   [[g] for g in probe["gold_id"]])
    probe["rank"] = ranks
    probe["rr"] = reciprocal_ranks(ranks)

    good = probe[probe.kind == "gold"]["rr"].values
    out = []
    for bad_kind, label in [("random", "C1 gold vs random"),
                            ("correct_answer", "C2 gold vs correct answer"),
                            ("perturbed_gold", "C3 gold vs perturbed (probe)")]:
        bad = probe[probe.kind == bad_kind]["rr"].values
        u, p = st.mannwhitneyu(good, bad, alternative="two-sided")
        out.append({"comparison": label,
                    "gold_mean_rr": round(float(good.mean()), 4),
                    "other_mean_rr": round(float(bad.mean()), 4),
                    "AUC": round(float(u / (len(good) * len(bad))), 4),
                    "p_value": f"{p:.3g}"})
    return pd.DataFrame(out)
