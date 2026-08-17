"""
Evaluation of generated distractors (Stage 2).

Metrics are computed per (arm, question) so that every comparison between
arms is PAIRED on the same questions.

Circularity note:
    Semantic scoring uses the PRETRAINED encoder (all-MiniLM-L6-v2), never the
    fine-tuned MNRL retriever. Scoring with the fine-tuned model would favour
    arms whose exemplars that same model retrieved, biasing the comparison.

Power note:
    Exact-match is binary and therefore low-power at n~100-190. The continuous
    similarity metrics are reported alongside it and carry the statistical
    weight; see compare_arms().
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# Normalisation and per-question metrics
# =============================================================================

_LATEX = re.compile(r"\\[a-zA-Z]+|[\\${}(),\s]")


def normalise_answer(s) -> str:
    """
    Normalise an answer string for exact matching.

    Strips LaTeX commands, delimiters, braces and whitespace, so that
    '\\( -5 \\)' and '-5' compare equal. Deliberately aggressive: the
    alternative (raw string equality) understates match rates because Eedi
    gold answers are LaTeX-wrapped while model output usually is not.
    """
    return _LATEX.sub("", str(s)).lower()


def exact_match(record: Dict) -> bool:
    """True if any generated candidate equals any gold distractor."""
    gold = {normalise_answer(g["distractor"]) for g in record["gold_distractors"]}
    gold.discard("")
    return any(normalise_answer(c["distractor"]) in gold
               for c in record.get("candidates", []))


def validity_flags(record: Dict) -> Dict:
    """Cheap quality checks that need no model."""
    cands = record.get("candidates", [])
    norm = [normalise_answer(c["distractor"]) for c in cands]
    correct = normalise_answer(record["correct_answer"])
    return {
        "n_candidates": len(cands),
        "collides_with_correct": any(n == correct for n in norm),
        "has_duplicates": len(set(norm)) < len(norm) if norm else False,
        "parse_failed": record.get("parse_status") == "failed",
    }


# =============================================================================
# Semantic metrics
# =============================================================================

def _encode(texts: Sequence[str], model) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    return model.encode(list(texts), normalize_embeddings=True,
                        show_progress_bar=False, batch_size=128)


def add_semantic_metrics(
    df: pd.DataFrame,
    records: List[Dict],
    model_name: str = "all-MiniLM-L6-v2",
) -> pd.DataFrame:
    """
    Add continuous similarity metrics, encoding every string in one pass.

    distractor_similarity
        max cosine( generated distractor , gold distractor ) over all pairs.
        Credits near-misses that exact-match discards.

    misconception_similarity
        max cosine( model's STATED misconception , gold misconception name ).
        This is the model's self-reported intent, not verified behaviour, and
        must be reported as such.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    texts, spans = [], []
    for r in records:
        cand_d = [c["distractor"] for c in r.get("candidates", [])]
        cand_m = [c.get("stated_misconception", "") for c in r.get("candidates", [])]
        gold_d = [g["distractor"] for g in r["gold_distractors"]]
        gold_m = [g["misconception_name"] for g in r["gold_distractors"]]
        start = len(texts)
        texts.extend(cand_d + cand_m + gold_d + gold_m)
        spans.append((start, len(cand_d), len(cand_m), len(gold_d), len(gold_m)))

    emb = _encode(texts, model)

    d_sim, m_sim = [], []
    for (start, nc, nm, ng, ngm) in spans:
        i = start
        cd = emb[i:i + nc]; i += nc
        cm = emb[i:i + nm]; i += nm
        gd = emb[i:i + ng]; i += ng
        gm = emb[i:i + ngm]
        d_sim.append(float((cd @ gd.T).max()) if len(cd) and len(gd) else np.nan)
        m_sim.append(float((cm @ gm.T).max()) if len(cm) and len(gm) else np.nan)

    df = df.copy()
    df["distractor_similarity"] = d_sim
    df["misconception_similarity"] = m_sim
    return df


def build_per_question_table(records_by_arm: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Assemble the per-(arm, question) metric table."""
    rows = []
    for arm, records in records_by_arm.items():
        for r in records:
            rows.append({
                "arm": arm,
                "question_id": r["question_id"],
                "exact_match": exact_match(r),
                "n_exemplars_matched": r.get("n_exemplars_matched", 0),
                **validity_flags(r),
            })
    return pd.DataFrame(rows)


# =============================================================================
# Paired comparison
# =============================================================================

def _bootstrap_ci(diff: np.ndarray, n_boot: int = 10000, seed: int = 0):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare_arms(
    df: pd.DataFrame,
    arm_a: str,
    arm_b: str,
    metrics: Sequence[str] = ("exact_match", "distractor_similarity",
                              "misconception_similarity"),
) -> pd.DataFrame:
    """
    Paired comparison of arm_b against arm_a on their common questions.

    Binary metrics use McNemar's exact test (the correct test for paired
    binary outcomes); continuous metrics use the Wilcoxon signed-rank test
    plus a bootstrap CI over questions. Positive deltas favour arm_b.
    """
    a = df[df.arm == arm_a].set_index("question_id")
    b = df[df.arm == arm_b].set_index("question_id")
    common = sorted(set(a.index) & set(b.index))
    if not common:
        return pd.DataFrame()
    a, b = a.loc[common], b.loc[common]

    rows = []
    for m in metrics:
        if m not in a.columns:
            continue
        va, vb = a[m].values, b[m].values
        ok = ~(pd.isna(va) | pd.isna(vb))
        va, vb = va[ok], vb[ok]
        if len(va) == 0:
            continue

        if set(np.unique(np.concatenate([va, vb]))) <= {0, 1, True, False}:
            va_i, vb_i = va.astype(int), vb.astype(int)
            n_b_only = int(((va_i == 1) & (vb_i == 0)).sum())
            n_a_only = int(((va_i == 0) & (vb_i == 1)).sum())
            disc = n_b_only + n_a_only
            p = stats.binomtest(n_a_only, disc, 0.5).pvalue if disc else 1.0
            test = f"McNemar exact (discordant={disc})"
            eff = float(vb_i.mean() - va_i.mean())
            lo, hi = _bootstrap_ci((vb_i - va_i).astype(float))
        else:
            diff = vb - va
            if np.allclose(diff, 0):
                p, test = 1.0, "Wilcoxon (all ties)"
            else:
                p = float(stats.wilcoxon(vb, va).pvalue)
                test = "Wilcoxon signed-rank"
            eff = float(diff.mean())
            lo, hi = _bootstrap_ci(diff)

        # Generic column names so comparisons with different arms concatenate
        # into one clean table rather than producing sparse arm-specific columns.
        rows.append({
            "comparison": f"{arm_b} vs {arm_a}",
            "metric": m,
            "n_paired": int(len(va)),
            "reference_mean": round(float(va.mean()), 4),
            "treatment_mean": round(float(vb.mean()), 4),
            "delta": round(eff, 4),
            "ci95_low": round(lo, 4),
            "ci95_high": round(hi, 4),
            "test": test,
            "p_value": f"{p:.4g}",
            "significant_0.05": "yes" if p < 0.05 else "no",
        })
    return pd.DataFrame(rows)


def holm_correct(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adj = [0.0] * n
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * p_values[i])
        adj[i] = min(1.0, running)
    return adj
