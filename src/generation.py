"""
Stage 2 — Retrieval-Augmented Distractor Generation (library).

Responsibilities:
    1. Collapse the QDP-level test split into question-level generation targets.
    2. Materialise Stage 1 retrieval indices into prompt-ready exemplar records.
    3. Assemble the retrieval-augmented prompt.
    4. Generate candidate distractors with a local instruction-tuned LLM.
    5. Parse model output robustly.

Design notes:
    - Stage 1 is frozen. This module only *reads* from it; the retriever, its
      training code, and its evaluation code are untouched.
    - The generation unit is the QUESTION, not the QDP. Under the deployment
      query representation (Subject | Construct | Question | Correct Answer)
      all QDPs of a question render to an identical query string, so the 431
      test QDPs collapse to 187 unique targets.
    - The query omits the distractor (unavailable at inference time) while
      retrieved corpus records retain it. That asymmetry is intentional: the
      exemplars supply exactly the misconception-bearing information the
      target question lacks.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Prompt template
# =============================================================================

SYSTEM_PROMPT = (
    "You are an experienced mathematics teacher who writes multiple-choice "
    "assessment items. A good distractor is not merely a wrong answer — it is "
    "the answer a student would actually arrive at by applying a specific, "
    "common misconception. Your task is to write distractors of that kind."
)

EXEMPLAR_BLOCK = (
    "### Example {i}\n"
    "Subject: {subject} | Construct: {construct}\n"
    "Question: {question}\n"
    "Correct answer: {correct_answer}\n"
    "Distractor: {distractor}\n"
)
# Appended only when misconception labels are shown (ablation switch).
EXEMPLAR_MISCONCEPTION_LINE = "Misconception encoded: {misconception}\n"

TARGET_BLOCK = (
    "### Target question\n"
    "Subject: {subject} | Construct: {construct}\n"
    "Question: {question}\n"
    "Correct answer: {correct_answer}\n"
)

INSTRUCTIONS = (
    "Write {m} distinct distractors for the target question. Each must:\n"
    "- be an answer a student could plausibly produce;\n"
    "- follow from one specific mathematical misconception, which you must name;\n"
    "- differ from the correct answer and from every other distractor;\n"
    "- match the notation and formatting style used in the question.\n\n"
    "Respond with JSON only, no commentary:\n"
    '{{"distractors": [{{"distractor": "...", "misconception": "..."}}]}}'
)


# =============================================================================
# 1. Generation targets
# =============================================================================

def build_question_targets(test_qdp: pd.DataFrame) -> List[Dict]:
    """
    Collapse the QDP-level test split into question-level generation targets.

    Each target carries its gold distractors (1-3 per question) so that the
    saved output is self-contained for later evaluation.

    Args:
        test_qdp: Test split QDPs (from scripts/01_prepare_dataset.py).

    Returns:
        List of target dicts, one per unique QuestionId.
    """
    targets = []
    for qid, grp in test_qdp.groupby("QuestionId", sort=True):
        first = grp.iloc[0]
        targets.append({
            "question_id": int(qid),
            "subject": first["SubjectName"],
            "construct": first["ConstructName"],
            "question_text": first["QuestionText"],
            "correct_answer": first["CorrectAnswerText"],
            "gold_distractors": [
                {
                    "distractor": r["DistractorText"],
                    "misconception_id": int(r["MisconceptionId"]),
                    "misconception_name": r["MisconceptionName"],
                }
                for _, r in grp.iterrows()
            ],
        })
    return targets


def build_query_frame(targets: List[Dict], template: str) -> pd.DataFrame:
    """
    Build the retrieval query frame for the targets.

    Rendered directly from the target fields with the deployment template, so
    the query text is exactly what Stage 1's Variant B ablation validated.
    """
    return pd.DataFrame([{
        "QuestionId": t["question_id"],
        "text": template.format(
            subject=t["subject"],
            construct=t["construct"],
            question=t["question_text"],
            correct_answer=t["correct_answer"],
            distractor="",  # absent from the deployment template
        ),
    } for t in targets])


# =============================================================================
# 2. Exemplar materialisation
# =============================================================================

def materialise_exemplars(
    retrieved_indices: Sequence[int],
    corpus_df: pd.DataFrame,
    k: int,
    exclude_question_id: Optional[int] = None,
) -> List[Dict]:
    """
    Turn Stage 1 retrieval indices into prompt-ready exemplar records.

    Args:
        retrieved_indices: Corpus row indices for one query, best-first.
        corpus_df: The retrieval corpus (train+val QDPs).
        k: Number of exemplars to keep.
        exclude_question_id: Drop exemplars from this question. Under the
            standard protocol the corpus (train+val) is disjoint from the test
            queries, so this is a no-op safety net — but it must be enforced
            because any overlap would leak a gold distractor into its own
            prompt.

    Returns:
        List of up to k exemplar dicts.
    """
    out: List[Dict] = []
    for rank, idx in enumerate(retrieved_indices, start=1):
        row = corpus_df.iloc[int(idx)]
        if exclude_question_id is not None and int(row["QuestionId"]) == exclude_question_id:
            continue
        out.append({
            "rank": rank,
            "corpus_index": int(idx),
            "question_id": int(row["QuestionId"]),
            "subject": row["SubjectName"],
            "construct": row["ConstructName"],
            "question": row["QuestionText"],
            "correct_answer": row["CorrectAnswerText"],
            "distractor": row["DistractorText"],
            "misconception_id": int(row["MisconceptionId"]),
            "misconception_name": row["MisconceptionName"],
        })
        if len(out) >= k:
            break
    return out


def annotate_exemplar_matches(exemplars: List[Dict], gold_misconception_ids: Sequence[int]) -> List[Dict]:
    """
    Flag which exemplars share a misconception with the target question.

    Recorded (never shown to the model) so that later analysis can test
    whether retrieval quality predicts generation quality — the mechanistic
    form of the Stage 2 research question.
    """
    gold = set(int(g) for g in gold_misconception_ids)
    for e in exemplars:
        e["is_misconception_match"] = e["misconception_id"] in gold
    return exemplars


# =============================================================================
# 3. Prompt construction
# =============================================================================

def build_prompt(
    target: Dict,
    exemplars: List[Dict],
    m: int = 5,
    show_misconception_labels: bool = True,
) -> Tuple[str, str]:
    """
    Assemble the retrieval-augmented prompt.

    Args:
        target: Target question dict from build_question_targets.
        exemplars: Prompt exemplars from materialise_exemplars.
        m: Number of distractor candidates to request.
        show_misconception_labels: If False, exemplar misconception names are
            withheld (labels-hidden ablation). Default True.

    Returns:
        (system_prompt, user_prompt)
    """
    parts = [
        "Below are real questions with teacher-written distractors."
        + (" Each example states the misconception that its distractor encodes.\n"
           if show_misconception_labels else "\n")
    ]
    for i, e in enumerate(exemplars, start=1):
        block = EXEMPLAR_BLOCK.format(
            i=i, subject=e["subject"], construct=e["construct"],
            question=e["question"], correct_answer=e["correct_answer"],
            distractor=e["distractor"],
        )
        if show_misconception_labels:
            block += EXEMPLAR_MISCONCEPTION_LINE.format(
                misconception=e["misconception_name"])
        parts.append(block)

    parts.append(TARGET_BLOCK.format(
        subject=target["subject"], construct=target["construct"],
        question=target["question_text"], correct_answer=target["correct_answer"],
    ))
    parts.append(INSTRUCTIONS.format(m=m))
    return SYSTEM_PROMPT, "\n".join(parts)


# =============================================================================
# 4. Response parsing
# =============================================================================

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
_PAIR = re.compile(
    r'"distractor"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"misconception"\s*:\s*"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)


def parse_response(raw: str, m: Optional[int] = None) -> Tuple[List[Dict], str]:
    """
    Parse candidates from the model's raw output.

    Strategy: strip markdown fences, attempt JSON on the outermost object,
    then fall back to a regex over distractor/misconception pairs. The raw
    string is always preserved by the caller, so a parser failure never loses
    a generation.

    Returns:
        (candidates, parse_status) where parse_status is 'json', 'regex',
        or 'failed'.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()

    match = _JSON_OBJ.search(text)
    if match:
        try:
            obj = json.loads(match.group(0))
            items = obj.get("distractors", []) if isinstance(obj, dict) else []
            cands = [
                {"distractor": str(d.get("distractor", "")).strip(),
                 "stated_misconception": str(d.get("misconception", "")).strip()}
                for d in items if isinstance(d, dict) and str(d.get("distractor", "")).strip()
            ]
            if cands:
                return (cands[:m] if m else cands), "json"
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

    cands = [
        {"distractor": d.strip(), "stated_misconception": mis.strip()}
        for d, mis in _PAIR.findall(text) if d.strip()
    ]
    if cands:
        return (cands[:m] if m else cands), "regex"
    return [], "failed"


# =============================================================================
# 5. LLM adapter
# =============================================================================

class QwenGenerator:
    """
    Local instruction-tuned generator (default Qwen2.5-7B-Instruct, 4-bit).

    Loaded once and reused across all targets. 4-bit quantisation keeps a 7B
    model within a 16 GB T4. Chosen over a hosted API for seed-controlled
    reproducibility, consistent with the standard used throughout Stage 1.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        seed: int = 42,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.seed = seed
        self._torch = torch

        print(f"  Loading generator: {model_name} (4bit={load_in_4bit})")
        kwargs = {"device_map": "auto"}
        if load_in_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()
        print("  Generator ready.")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Run one chat completion and return the assistant's raw text."""
        from transformers import set_seed
        set_seed(self.seed)

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=0.9 if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


# =============================================================================
# 6. Resume support
# =============================================================================

def load_completed_question_ids(path) -> set:
    """
    Read already-generated question ids from a JSONL file.

    Enables recovery after a Colab disconnect: completed targets are skipped
    on the next run rather than regenerated.
    """
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return set()
    done = set()
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["question_id"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue  # tolerate a truncated final line from a hard kill
    return done
