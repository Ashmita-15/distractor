"""
Stage 3 — Distractor re-ranking and selection.

Pipeline position: Stage 2 emits M candidate distractors per question; this
module filters invalid ones and selects the best.

Measured headroom (Stage 2 deployed run, 187 questions):
    take first valid candidate   28.9% exact match
    oracle best-of-M (ceiling)   49.7%
    => 20.9 percentage points available to a ranker

Two-part separation:
    HARD FILTER (deterministic)  correctness constraints — empty, equal to the
        correct answer, duplicate. These are not quality judgements and no
        model is asked to relitigate them.
    SOFT RANKING (heuristic)     plausibility judgements over surviving
        candidates.

Circularity constraints:
    - The Stage 1 MNRL retriever is never used to score candidates.
    - Gold distractors are used ONLY by the oracle strategy (R4), which is a
      diagnostic ceiling, never a deployable system.
    - The heuristic uses no gold information and no tuned parameters: the
      three features are equally weighted by fiat, not fitted, so no test-set
      tuning can occur.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.gen_evaluation import normalise_answer

_NUM = re.compile(r"-?\d+\.?\d*")

STRATEGIES = ("R0_first", "R1_random", "R2_heuristic", "R4_oracle")
STRATEGY_R3 = "R3_llm"
STRATEGY_DESCRIPTIONS = {
    "R0_first": "First valid candidate (Stage 2 default; baseline)",
    "R1_random": "Random valid candidate (controls for 'any selection != first')",
    "R2_heuristic": "Heuristic plausibility ranking (no gold, no tuning)",
    "R3_llm": "LLM-judge semantic ranking (validated pointwise judge)",
    "R4_oracle": "ORACLE — best candidate by gold similarity (ceiling, diagnostic)",
}


def judge_score_key(question_id: int, candidate: str) -> str:
    """
    Stable key for a cached judge score.

    Keyed on the NORMALISED candidate text rather than a list index: the hard
    filter removes duplicates, so normalised text is unique within a question,
    and the key survives any change to filtering order.
    """
    return f"{int(question_id)}||{normalise_answer(candidate)}"


# =============================================================================
# Hard filter (deterministic)
# =============================================================================

def hard_filter(record: Dict) -> Tuple[List[Dict], Dict]:
    """
    Remove candidates that are invalid as distractors.

    Rejects, in order: empty/whitespace output; normalised equality with the
    correct answer (it is not a distractor); duplicates within the candidate
    set (they add no diagnostic value). Order of surviving candidates is
    preserved, so the first survivor is the R0 baseline.

    Returns:
        (valid_candidates, filter_stats)
    """
    correct = normalise_answer(record["correct_answer"])
    seen, valid = set(), []
    stats = {"n_input": 0, "n_empty": 0, "n_correct_answer": 0,
             "n_duplicate": 0, "n_valid": 0}

    for c in record.get("candidates", []):
        stats["n_input"] += 1
        text = str(c.get("distractor", "")).strip()
        norm = normalise_answer(text)
        if not text:
            stats["n_empty"] += 1
            continue
        if norm == correct:
            stats["n_correct_answer"] += 1
            continue
        if norm in seen:
            stats["n_duplicate"] += 1
            continue
        seen.add(norm)
        valid.append(dict(c, distractor=text))

    stats["n_valid"] = len(valid)
    return valid, stats


# =============================================================================
# Heuristic features (no gold, no tuning)
# =============================================================================

def _number(s) -> Optional[float]:
    m = _NUM.search(str(s).replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def heuristic_score(candidate: str, correct_answer: str) -> Dict[str, float]:
    """
    Plausibility score in [0, 1] from three equally weighted features.

    type_match      A distractor should be the same kind of object as the
                    correct answer: numeric answers invite numeric errors.
    magnitude       For numeric pairs, a student's miscalculation usually
                    lands in the same ballpark; wildly different magnitudes
                    are implausible. Neutral (0.5) when not both numeric.
    length_match    Answers far longer than the correct one are usually
                    explanations or malformed output rather than answers.

    Weights are fixed at 1/3 each BY FIAT, not fitted. Nothing here is tuned
    on the evaluation set, so the strategy cannot leak test information.
    """
    c_num, a_num = _number(candidate), _number(correct_answer)
    both_num = (c_num is not None) and (a_num is not None)
    type_match = 1.0 if ((c_num is None) == (a_num is None)) else 0.0

    if both_num and a_num != 0 and c_num != 0:
        ratio = abs(c_num) / abs(a_num)
        magnitude = 1.0 / (1.0 + abs(np.log10(ratio)))
    elif both_num:
        magnitude = 0.5          # a zero on either side makes the ratio undefined
    else:
        magnitude = 0.5

    la, lc = max(len(str(correct_answer)), 1), len(str(candidate))
    length_match = 1.0 / (1.0 + abs(lc - la) / la)

    total = (type_match + magnitude + length_match) / 3.0
    return {"type_match": type_match, "magnitude": magnitude,
            "length_match": length_match, "heuristic_score": total}


# =============================================================================
# Selection strategies
# =============================================================================

def _gold_similarity(candidate: str, gold_distractors: Sequence[Dict]) -> float:
    """
    Oracle-only signal: exact match first, else an order-aware string ratio.

    Used solely by R4 to establish the ceiling. Never available to a
    deployable strategy.

    Order-aware similarity is essential for numeric answers. Character-set
    overlap (Jaccard) scores '30' and '300' as identical, because sets discard
    both order and multiplicity — which caused the oracle to pass over a true
    gold match in favour of a same-digits impostor. SequenceMatcher respects
    both, so an exact match always strictly outranks a near-miss.
    """
    from difflib import SequenceMatcher

    cn = normalise_answer(candidate)
    golds = [normalise_answer(g["distractor"]) for g in gold_distractors]
    golds = [g for g in golds if g]
    if not cn or not golds:
        return 0.0
    if cn in golds:
        return 1.0
    # strictly below 1.0 so exact matches always win
    return max(SequenceMatcher(None, cn, g).ratio() for g in golds) * 0.999


def select(
    strategy: str,
    record: Dict,
    valid: List[Dict],
    rng: Optional[np.random.RandomState] = None,
    k: int = 1,
    judge_scores: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """
    Select up to k candidates under the given strategy.

    All strategies operate on the SAME hard-filtered candidate list, so the
    only thing that varies between them is the selection rule.
    """
    if not valid:
        return []
    if strategy == "R0_first":
        ranked = list(valid)
    elif strategy == "R1_random":
        rng = rng or np.random.RandomState(0)
        ranked = [valid[i] for i in rng.permutation(len(valid))]
    elif strategy == "R2_heuristic":
        scored = [(heuristic_score(c["distractor"], record["correct_answer"])
                   ["heuristic_score"], i, c) for i, c in enumerate(valid)]
        # -score then original index: ties resolve to generation order, so R2
        # degenerates gracefully to R0 rather than to an arbitrary permutation
        ranked = [c for _, _, c in sorted(scored, key=lambda t: (-t[0], t[1]))]
    elif strategy == STRATEGY_R3:
        if judge_scores is None:
            raise ValueError("R3_llm requires judge_scores; run "
                             "scripts/16_score_candidates.py first")
        # Unscored candidates (parse failures) sort last but stay selectable,
        # so a question is never dropped for lack of a score.
        scored = [(judge_scores.get(judge_score_key(record["question_id"],
                                                    c["distractor"]), -1.0), i, c)
                  for i, c in enumerate(valid)]
        # -score then original index: ties resolve to generator order, which
        # Experiment 3.1 showed is itself an informative signal (R0 > R1).
        ranked = [c for _, _, c in sorted(scored, key=lambda t: (-t[0], t[1]))]
    elif strategy == "R4_oracle":
        scored = [(_gold_similarity(c["distractor"], record["gold_distractors"]), i, c)
                  for i, c in enumerate(valid)]
        ranked = [c for _, _, c in sorted(scored, key=lambda t: (-t[0], t[1]))]
    else:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of "
                         f"{STRATEGIES + (STRATEGY_R3,)}")

    return ranked[:k] if k == 1 else _diverse_top_k(ranked, k)


def _diverse_top_k(ranked: List[Dict], k: int, min_jaccard_gap: float = 0.9) -> List[Dict]:
    """
    Greedy top-k that skips near-duplicates of already-selected candidates.

    Exact duplicates are already removed by the hard filter; this additionally
    avoids trivially similar variants (e.g. '12' and '12.0'), which would
    waste distractor slots without adding diagnostic value.
    """
    chosen: List[Dict] = []
    for c in ranked:
        cn = normalise_answer(c["distractor"])
        too_similar = False
        for s in chosen:
            sn = normalise_answer(s["distractor"])
            union = len(set(cn) | set(sn)) or 1
            if len(set(cn) & set(sn)) / union >= min_jaccard_gap:
                too_similar = True
                break
        if not too_similar:
            chosen.append(c)
        if len(chosen) >= k:
            break
    return chosen
