#!/usr/bin/env python3
"""
Stage 11 — Metric sensitivity validation (positive control).

Question
    Can Exact Match distinguish a good distractor from a bad one? If it
    cannot, the Experiment 2.1 "oracle == random" null is an artefact of the
    instrument rather than a property of the pipeline.

Method — conditions with KNOWN quality, scored by each candidate metric:

    T1  gold_i scored against ALL golds          -> known TRUE POSITIVE
    T2  gold_i scored against golds MINUS gold_i -> known GOOD but not in
                                                    the reference set
    T3  random corpus distractor                 -> known BAD (irrelevant)
    T4  the correct answer                       -> known INVALID
    T5  digit-perturbed gold                     -> known NEAR-MISS

    T2 is the crux. It is a teacher-written distractor for that exact
    question, so it is unambiguously good; it simply is not the particular
    gold string being compared against. A metric that scores T2 no higher
    than T3 cannot tell an excellent distractor from an irrelevant one, and
    therefore cannot support a null result.

Discrimination is quantified as AUC (probability that a random T2 outscores a
random T3) with a Mann-Whitney U test. AUC ~ 0.5 means no discrimination.

Usage
    python scripts/11_metric_sensitivity.py
"""

import sys
import json
import re
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from src.config import RESULTS_DIR, OUTPUT_DIR, RANDOM_SEED
from src.utils import set_seed, print_section_header
from src.retrieval_experiment import load_corpus_and_queries
from src.generation import build_question_targets
from src.gen_evaluation import normalise_answer

OUT_DIR = RESULTS_DIR / "metric_sensitivity"
GEN_DIR = OUTPUT_DIR / "generation" / "exp21"

_NUM = re.compile(r"-?\d+\.?\d*")


def extract_number(s):
    """First number in a string, or None. Used for numeric proximity."""
    m = _NUM.search(str(s).replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def perturb(s):
    """Digit-perturb a numeric answer, leaving non-numeric strings unchanged."""
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


def em(candidate, references) -> float:
    """Exact match (normalised) of one candidate against a reference set."""
    refs = {normalise_answer(r) for r in references}
    refs.discard("")
    return float(normalise_answer(candidate) in refs)


def numeric_proximity(candidate, references, correct):
    """
    1 - relative error to the closest reference, scaled by the correct answer.
    None when the strings are not numeric.
    """
    c = extract_number(candidate)
    scale = abs(extract_number(correct) or 1.0) or 1.0
    vals = [extract_number(r) for r in references]
    vals = [v for v in vals if v is not None]
    if c is None or not vals:
        return None
    return float(max(0.0, 1.0 - min(abs(c - v) for v in vals) / scale))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    set_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    print_section_header("METRIC SENSITIVITY — POSITIVE CONTROL")

    corpus_df, test_qdp, _ = load_corpus_and_queries("test")
    targets = build_question_targets(test_qdp)
    multi = [t for t in targets if len(t["gold_distractors"]) >= 2]
    print(f"  test questions: {len(targets)} | with >=2 gold distractors: {len(multi)}")
    print("  (T1/T2 require >=2 golds so one can be held out)")

    corpus_distractors = corpus_df["DistractorText"].tolist()

    rows = []
    for t in multi:
        golds = [g["distractor"] for g in t["gold_distractors"]]
        correct = t["correct_answer"]
        for i, gi in enumerate(golds):
            others = [g for j, g in enumerate(golds) if j != i]
            rand_d = corpus_distractors[rng.randint(len(corpus_distractors))]
            for cond, cand, refs in [
                ("T1_gold_in_ref", gi, golds),
                ("T2_good_not_in_ref", gi, others),
                ("T3_random_distractor", rand_d, others),
                ("T4_correct_answer", correct, others),
                ("T5_perturbed_gold", perturb(gi), others),
            ]:
                rows.append({
                    "question_id": t["question_id"], "gold_idx": i,
                    "condition": cond, "candidate": str(cand),
                    "references": json.dumps([str(r) for r in refs]),
                    "correct_answer": str(correct),
                    "exact_match": em(cand, refs),
                    "numeric_proximity": numeric_proximity(cand, refs, correct),
                })
    df = pd.DataFrame(rows)
    print(f"  evaluation instances: {len(df)} ({df.condition.nunique()} conditions)")

    # ---- Embedding similarity (pretrained encoder; never the fine-tuned one) ----
    print("\n  Encoding for embedding similarity (pretrained MiniLM)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    cands = df["candidate"].tolist()
    ref_lists = [json.loads(r) for r in df["references"]]
    flat_refs, spans = [], []
    for r in ref_lists:
        spans.append((len(flat_refs), len(r)))
        flat_refs.extend(r)
    ce = model.encode(cands, normalize_embeddings=True, show_progress_bar=False, batch_size=256)
    re_ = model.encode(flat_refs, normalize_embeddings=True, show_progress_bar=False, batch_size=256)
    df["embedding_similarity"] = [
        float((ce[i:i+1] @ re_[s:s+n].T).max()) if n else np.nan
        for i, (s, n) in enumerate(spans)
    ]

    METRICS = ["exact_match", "embedding_similarity", "numeric_proximity"]

    print_section_header("Condition means")
    summ = df.groupby("condition")[METRICS].mean().round(4)
    summ["n"] = df.groupby("condition").size()
    print(summ.to_string())
    summ.to_csv(OUT_DIR / "sensitivity_conditions.csv")

    # ---- Discrimination: can each metric separate GOOD (T2) from BAD (T3)? ----
    print_section_header("DISCRIMINATION — T2 (good) vs T3 (bad)")
    print("  AUC = P(a random good distractor outscores a random bad one).")
    print("  0.5 = no discrimination; 1.0 = perfect.\n")
    disc_rows = []
    for m in METRICS:
        for good_c, bad_c, label in [("T2_good_not_in_ref", "T3_random_distractor", "T2 vs T3 (good vs random)"),
                                     ("T2_good_not_in_ref", "T4_correct_answer", "T2 vs T4 (good vs correct-answer)"),
                                     ("T2_good_not_in_ref", "T5_perturbed_gold", "T2 vs T5 (good vs near-miss)")]:
            a = df[df.condition == good_c][m].dropna().values
            b = df[df.condition == bad_c][m].dropna().values
            if len(a) == 0 or len(b) == 0:
                continue
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            auc = u / (len(a) * len(b))
            disc_rows.append({
                "metric": m, "comparison": label,
                "good_mean": round(float(a.mean()), 4),
                "bad_mean": round(float(b.mean()), 4),
                "AUC": round(float(auc), 4),
                "p_value": f"{p:.3g}",
                "discriminates": "YES" if (p < 0.05 and auc > 0.6) else "NO",
            })
    disc = pd.DataFrame(disc_rows)
    print(disc.to_string(index=False))
    disc.to_csv(OUT_DIR / "sensitivity_discrimination.csv", index=False)

    # ---- Headline: exact-match false-negative rate on known-good distractors ----
    t2 = df[df.condition == "T2_good_not_in_ref"]
    fn = 1.0 - t2["exact_match"].mean()
    print_section_header("HEADLINE")
    print(f"  Exact Match credits a KNOWN-GOOD distractor (T2) "
          f"{t2['exact_match'].mean():.1%} of the time.")
    print(f"  => false-negative rate on genuinely good distractors: {fn:.1%}")

    em_auc = disc[(disc.metric == "exact_match") &
                  (disc.comparison.str.startswith("T2 vs T3"))]["AUC"]
    emb_auc = disc[(disc.metric == "embedding_similarity") &
                   (disc.comparison.str.startswith("T2 vs T3"))]["AUC"]
    em_auc = float(em_auc.iloc[0]) if len(em_auc) else float("nan")
    emb_auc = float(emb_auc.iloc[0]) if len(emb_auc) else float("nan")
    print(f"  Exact Match  AUC (good vs random): {em_auc:.3f}")
    print(f"  Embedding    AUC (good vs random): {emb_auc:.3f}")

    write_report(summ, disc, fn, em_auc, emb_auc)
    print("\n  ✅ Sensitivity validation complete.")


def write_report(summ, disc, fn, em_auc, emb_auc) -> None:
    def md(f, index=True):
        try:
            return f.to_markdown(index=index)
        except ImportError:
            return "```\n" + f.to_string(index=index) + "\n```"

    if em_auc < 0.55:
        verdict = (
            f"**Exact Match cannot distinguish good distractors from bad ones "
            f"(AUC = {em_auc:.3f}, chance = 0.5).** It fails to credit a known-good "
            f"distractor {fn:.1%} of the time, scoring it identically to an "
            f"irrelevant one. A metric with this profile cannot support a null "
            f"result: the Experiment 2.1 'oracle == random' finding is an "
            f"ARTEFACT OF THE INSTRUMENT and must be withdrawn pending "
            f"re-evaluation with a sensitive metric.")
    elif em_auc < 0.7:
        verdict = (
            f"**Exact Match discriminates weakly (AUC = {em_auc:.3f}).** It misses "
            f"{fn:.1%} of known-good distractors. Treat the Experiment 2.1 null as "
            f"provisional and confirm it with a more sensitive metric before "
            f"drawing conclusions.")
    else:
        verdict = (
            f"**Exact Match discriminates acceptably (AUC = {em_auc:.3f}).** "
            f"The Experiment 2.1 'oracle == random' null is a genuine property of "
            f"the pipeline, not an instrument failure.")

    lines = [
        "# Metric Sensitivity Validation (Positive Control)", "",
        "Whether Exact Match can separate good distractors from bad ones. If it "
        "cannot, the Experiment 2.1 null is uninterpretable.", "",
        "## Conditions", "",
        "| Condition | Candidate | Known quality |",
        "|---|---|---|",
        "| T1 | gold, scored against all golds | true positive (sanity) |",
        "| T2 | gold, scored against the OTHER golds | **good, but absent from the reference set** |",
        "| T3 | random corpus distractor | bad |",
        "| T4 | the correct answer | invalid |",
        "| T5 | digit-perturbed gold | near-miss |",
        "",
        "T2 is the crux: a teacher-written distractor for that exact question, "
        "hence unambiguously good, that simply is not the reference string.",
        "", "## Condition means", "", md(summ), "",
        "## Discrimination", "", md(disc, index=False), "",
        "## Verdict", "", verdict, "",
        f"For contrast, embedding similarity achieves AUC = {emb_auc:.3f} on the "
        f"same good-vs-bad comparison, so a more sensitive instrument is available.",
        "", "---", "*Auto-generated by scripts/11_metric_sensitivity.py.*",
    ]
    p = OUT_DIR / "metric_sensitivity_report.md"
    p.write_text("\n".join(lines))
    print(f"  [Saved] {p}")


if __name__ == "__main__":
    main()
