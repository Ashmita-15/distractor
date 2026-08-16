#!/usr/bin/env python3
"""
Stage 07 — Query-representation ablation.

Research question
    Do the retrieval gains of the MNRL retriever persist when the query is
    restricted to information that will actually be available at generation
    time (Stage 2), where the distractor does not yet exist?

Design
    ONE independent variable: the query text representation. Held fixed:
    dataset, split, corpus, model checkpoint, retrieval algorithm, metrics,
    and evaluation code. No retraining — the same checkpoint is re-evaluated
    under three query representations, so any difference is attributable to
    the query representation alone.

    The corpus is encoded ONCE and reused byte-identically across all
    variants. This is deployment-faithful: historical corpus items have known
    distractors, an incoming question does not, so the deployed setting is
    genuinely asymmetric.

Variants (query side only; corpus always "full")
    C  full            Subject + Construct + Question + Correct Answer + Distractor  (reference)
    B  no_distractor   Subject + Construct + Question + Correct Answer
    A  question_answer Question + Correct Answer                                     (deployment-faithful)

Supplementary diagnostic (--symmetric)
    Also evaluates A and B with the corpus rendered in the SAME reduced
    template. This separates two causes of any drop: information loss
    (distractor genuinely carried signal) vs. asymmetry / train-inference
    format mismatch. Reported separately from the primary comparison.

Usage
    python scripts/07_query_ablation.py --model outputs/models/finetuned_mnrl
    python scripts/07_query_ablation.py --model ... --baseline --symmetric
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    QDP_TEXT_TEMPLATES, RESULTS_DIR, EMBEDDINGS_DIR, FIGURES_DIR,
    BASELINE_MODEL_NAME,
)
from src.utils import set_seed, print_section_header, setup_matplotlib
from src.retrieval_experiment import load_corpus_and_queries
from src.baseline_retrieval import encode_texts

K_VALUES = [10, 25]
METRIC_COLS = ["HitRate@10", "Recall@10", "MRR@10", "nDCG@10",
               "HitRate@25", "Recall@25", "MRR@25", "nDCG@25"]
KEYMAP = {"HitRate": "hit_rate", "Recall": "recall", "MRR": "mrr", "nDCG": "ndcg"}

# (label, variant_key, description) — ordered as reported
VARIANTS = [
    ("C", "full", "Subject + Construct + Question + Correct Answer + Distractor"),
    ("B", "no_distractor", "Subject + Construct + Question + Correct Answer"),
    ("A", "question_answer", "Question + Correct Answer"),
]

OUT_DIR = RESULTS_DIR / "query_ablation"
FIG_DIR = FIGURES_DIR / "query_ablation"


def public_metrics(metrics: dict) -> dict:
    """Convert internal metric keys to the public column names."""
    return {f"{p}@{k}": metrics[f"{KEYMAP[p]}@{k}"]
            for k in K_VALUES for p in ["HitRate", "Recall", "MRR", "nDCG"]}


def evaluate(model_path, corpus_df, query_df, corpus_emb, tag):
    """
    Encode queries and score retrieval against a fixed, precomputed corpus.

    Reuses the project's retrieval + metric implementations unchanged; only
    the query text differs between calls.
    """
    from src.baseline_retrieval import build_faiss_index, retrieve_top_k
    from src.evaluation import compute_retrieval_metrics

    query_emb = encode_texts(query_df["text"].tolist(),
                             model_name_or_path=str(model_path))
    index = build_faiss_index(corpus_emb)
    _, indices = retrieve_top_k(
        query_embeddings=query_emb, index=index, k=50,
        query_question_ids=query_df["QuestionId"].values,
        corpus_question_ids=corpus_df["QuestionId"].values,
    )
    corpus_misc = corpus_df["MisconceptionId"].values
    query_misc = query_df["MisconceptionId"].values
    metrics = compute_retrieval_metrics(
        query_misconception_ids=query_misc,
        retrieved_misconception_ids=corpus_misc[indices],
        corpus_misconception_ids=corpus_misc,
        k_values=K_VALUES,
        queries_in_corpus=False,
    )
    np.save(OUT_DIR / f"{tag}_indices.npy", indices)
    return metrics, indices


def per_query_rr(indices, corpus_df, query_df, k=10):
    """Per-query reciprocal rank@k, for paired significance testing."""
    cm = corpus_df["MisconceptionId"].values
    qm = query_df["MisconceptionId"].values
    rel = (cm[indices[:, :k]] == qm[:, None])
    rr = np.zeros(len(rel))
    for i, r in enumerate(rel):
        w = np.where(r)[0]
        rr[i] = 1.0 / (w[0] + 1) if len(w) else 0.0
    return rr, rel.any(axis=1).astype(float)


def main() -> None:
    ap = argparse.ArgumentParser(description="Query-representation ablation")
    ap.add_argument("--model", type=str, required=True,
                    help="Path to the trained MNRL model directory")
    ap.add_argument("--query-split", choices=["test", "val"], default="test")
    ap.add_argument("--baseline", action="store_true",
                    help="Also evaluate the pretrained baseline under each variant")
    ap.add_argument("--symmetric", action="store_true",
                    help="Supplementary diagnostic: also render the CORPUS with the "
                         "reduced template (separates information loss from asymmetry)")
    args = ap.parse_args()

    set_seed()
    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print_section_header("QUERY-REPRESENTATION ABLATION")
    print(f"  Model: {args.model}")
    print(f"  Query split: {args.query_split}  |  corpus template: full (fixed)")

    corpus_df, query_df_base, _ = load_corpus_and_queries(args.query_split)

    # ---- Encode the corpus ONCE; reused identically by every variant ----
    print_section_header("Encoding corpus (once, shared by all variants)")
    corpus_emb = encode_texts(corpus_df["text"].tolist(),
                              model_name_or_path=str(args.model))
    np.save(EMBEDDINGS_DIR / "ablation_corpus_embeddings.npy", corpus_emb)

    from src.data_loader import create_text_representation

    rows, per_query = [], {}
    for label, key, desc in VARIANTS:
        print_section_header(f"Variant {label} — {key}")
        print(f"  {desc}")
        qdf = create_text_representation(query_df_base, template=QDP_TEXT_TEMPLATES[key])
        print(f"  Example query: {qdf['text'].iloc[0][:170]!r}")
        m, idx = evaluate(args.model, corpus_df, qdf, corpus_emb, f"mnrl_{key}")
        rr, hit = per_query_rr(idx, corpus_df, qdf)
        per_query[label] = {"rr": rr, "hit": hit}
        rows.append({"variant": label, "query_representation": key,
                     "fields": desc, "model": "MNRL", **public_metrics(m)})

    # ---- Optional: pretrained baseline under the same variants ----
    if args.baseline:
        print_section_header("Pretrained baseline under each variant")
        base_corpus_emb = encode_texts(corpus_df["text"].tolist(),
                                       model_name_or_path=BASELINE_MODEL_NAME)
        for label, key, desc in VARIANTS:
            qdf = create_text_representation(query_df_base,
                                             template=QDP_TEXT_TEMPLATES[key])
            m, _ = evaluate(BASELINE_MODEL_NAME, corpus_df, qdf,
                            base_corpus_emb, f"baseline_{key}")
            rows.append({"variant": label, "query_representation": key,
                         "fields": desc, "model": "Baseline", **public_metrics(m)})

    # ---- Optional supplementary diagnostic: symmetric reduced templates ----
    if args.symmetric:
        print_section_header("SUPPLEMENTARY: symmetric (corpus also reduced)")
        print("  Diagnostic only — separates information loss from asymmetry.")
        for label, key, desc in VARIANTS:
            if key == "full":
                continue
            cdf = create_text_representation(corpus_df, template=QDP_TEXT_TEMPLATES[key])
            qdf = create_text_representation(query_df_base, template=QDP_TEXT_TEMPLATES[key])
            c_emb = encode_texts(cdf["text"].tolist(), model_name_or_path=str(args.model))
            m, _ = evaluate(args.model, cdf, qdf, c_emb, f"mnrl_sym_{key}")
            rows.append({"variant": f"{label}-sym", "query_representation": key,
                         "fields": f"[symmetric corpus] {desc}", "model": "MNRL",
                         **public_metrics(m)})

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "query_ablation_results.csv"
    df.to_csv(csv_path, index=False)
    print_section_header("RESULTS")
    print(df[["variant", "model", "query_representation"] + METRIC_COLS].to_string(index=False))
    print(f"\n  [Saved] {csv_path}")

    # ---- Paired significance vs. the reference variant C ----
    from scipy import stats as st
    sig_rows = []
    ref = per_query["C"]
    for label in ["B", "A"]:
        d = per_query[label]["rr"] - ref["rr"]
        rng = np.random.RandomState(0)
        bs = d[rng.randint(0, len(d), size=(10000, len(d)))].mean(axis=1)
        lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        b = int(((ref["hit"] == 1) & (per_query[label]["hit"] == 0)).sum())
        c = int(((ref["hit"] == 0) & (per_query[label]["hit"] == 1)).sum())
        p = st.binomtest(c, b + c, 0.5).pvalue if b + c else 1.0
        sig_rows.append({"comparison": f"{label} vs C (reference)",
                         "delta_MRR@10": round(float(d.mean()), 4),
                         "ci95_low": round(float(lo), 4), "ci95_high": round(float(hi), 4),
                         "mcnemar_p_hit@10": f"{p:.4g}",
                         "significant_at_0.05": "yes" if p < 0.05 else "no"})
    sig = pd.DataFrame(sig_rows)
    sig.to_csv(OUT_DIR / "query_ablation_significance.csv", index=False)
    print("\n  Paired per-query tests vs. reference variant C (two-sided):")
    print(sig.to_string(index=False))

    # ---- Figure ----
    mnrl = df[(df.model == "MNRL") & (~df.variant.str.contains("-sym"))]
    show = ["HitRate@10", "Recall@10", "MRR@10", "nDCG@10"]
    order = ["C", "B", "A"]
    mnrl = mnrl.set_index("variant").loc[order].reset_index()
    x, w = np.arange(len(show)), 0.26
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (lab, colour) in enumerate(zip(order, ["#2563eb", "#f59e0b", "#dc2626"])):
        vals = [mnrl[mnrl.variant == lab][m].iloc[0] for m in show]
        ax.bar(x + (i - 1) * w, vals, w, label=f"Variant {lab}", color=colour,
               edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(show)
    ax.set_ylabel("metric value")
    ax.set_title("Retrieval performance by query representation\n"
                 "(same MNRL checkpoint, same corpus, same evaluation)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "query_ablation.png", dpi=200, bbox_inches="tight")
    print(f"\n  [Saved] {FIG_DIR / 'query_ablation.png'}")

    write_report(df, sig)
    print("\n  ✅ Query-representation ablation complete.")


def write_report(df: pd.DataFrame, sig: pd.DataFrame) -> None:
    """Auto-generate the interpretation report."""
    mnrl = df[(df.model == "MNRL") & (~df.variant.str.contains("-sym"))].set_index("variant")
    C, B, A = mnrl.loc["C"], mnrl.loc["B"], mnrl.loc["A"]
    dA = A["MRR@10"] - C["MRR@10"]
    retention = A["MRR@10"] / C["MRR@10"] * 100 if C["MRR@10"] else float("nan")

    base = df[df.model == "Baseline"]
    base_note = ""
    if len(base):
        bC = base[base.variant == "C"]["MRR@10"].iloc[0]
        bA = base[base.variant == "A"]["MRR@10"].iloc[0]
        beats = A["MRR@10"] > bA
        base_note = (
            f"\nUnder the deployment-faithful query (Variant A), the fine-tuned "
            f"retriever scores MRR@10 = {A['MRR@10']:.4f} against a pretrained "
            f"baseline of {bA:.4f} evaluated on the *same* query representation "
            f"— fine-tuning {'still helps' if beats else 'no longer helps'} in the "
            f"deployment setting. (Baseline under the full query was {bC:.4f}.)\n")

    if retention >= 90:
        verdict = ("**Gains persist.** The deployment-faithful query retains "
                   f"{retention:.1f}% of reference MRR@10. The current retriever can be "
                   "carried into Stage 2 with question-only queries; retraining is not "
                   "required on this evidence.")
    elif retention >= 70:
        verdict = ("**Partial degradation.** The deployment-faithful query retains "
                   f"{retention:.1f}% of reference MRR@10. The retriever remains usable "
                   "but a meaningful share of its performance depended on the distractor. "
                   "Retraining on question-only inputs is advisable and likely to recover "
                   "part of the gap.")
    else:
        verdict = ("**Substantial degradation.** The deployment-faithful query retains "
                   f"only {retention:.1f}% of reference MRR@10. Most of the reported "
                   "performance depended on information unavailable at generation time. "
                   "Retraining with question-only inputs is required before Stage 2 "
                   "integration, and Stage 1 headline numbers must be reported with this "
                   "caveat attached.")

    lines = [
        "# Query-Representation Ablation — Report", "",
        "One independent variable: the query text representation. The model "
        "checkpoint, corpus (encoded once and reused byte-identically), split, "
        "retrieval algorithm, metrics, and evaluation code are all fixed. "
        "No retraining was performed.", "",
        "## Variants", "",
        "| Variant | Query fields | MRR@10 | HitRate@10 | nDCG@10 | Recall@10 |",
        "|---|---|---|---|---|---|",
        f"| C (reference) | {C['fields']} | {C['MRR@10']:.4f} | {C['HitRate@10']:.4f} | {C['nDCG@10']:.4f} | {C['Recall@10']:.4f} |",
        f"| B | {B['fields']} | {B['MRR@10']:.4f} | {B['HitRate@10']:.4f} | {B['nDCG@10']:.4f} | {B['Recall@10']:.4f} |",
        f"| A (deployment) | {A['fields']} | {A['MRR@10']:.4f} | {A['HitRate@10']:.4f} | {A['nDCG@10']:.4f} | {A['Recall@10']:.4f} |",
        "",
        "## Effect of removing the distractor", "",
        f"- Variant B (metadata kept, distractor removed): ΔMRR@10 = {B['MRR@10']-C['MRR@10']:+.4f}",
        f"- Variant A (deployment-faithful): ΔMRR@10 = {dA:+.4f} "
        f"({retention:.1f}% of reference retained)",
        f"- Metadata contribution (A → B): {B['MRR@10']-A['MRR@10']:+.4f} MRR@10",
        "",
        "### Paired per-query significance vs. reference", "",
        sig.to_string(index=False), "",
        base_note,
        "## Verdict", "", verdict, "",
        "## Interpretation caveat", "",
        "A drop under Variant A has two possible causes that this ablation "
        "cannot separate: (i) genuine information loss — the distractor carried "
        "real misconception signal — and (ii) train/inference mismatch, since the "
        "model was trained on full-QDP text and has never seen question-only "
        "input. If the `--symmetric` diagnostic was run, compare A against A-sym: "
        "a large gap points to asymmetry/format mismatch (favouring retraining), "
        "while a small gap points to genuine information loss (favouring a change "
        "of task framing).",
        "",
        "---",
        "*Auto-generated by scripts/07_query_ablation.py.*",
    ]
    p = OUT_DIR / "query_ablation_report.md"
    p.write_text("\n".join(lines))
    print(f"  [Saved] {p}")


if __name__ == "__main__":
    main()
