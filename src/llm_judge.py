"""
LLM-as-judge for distractor quality (Stage 2 evaluation).

Substitutes for human evaluation, which is unavailable, on the two constructs
no automatic metric can measure: plausibility and educational usefulness. It
also targets the near-miss blind spot — embedding similarity, numeric
proximity, and exact match all fail to separate a good distractor from a
digit-perturbed one (AUC ~0.56).

Validity without human labels
    A judge is normally validated by correlating it against human ratings.
    Instead this module validates against KNOWN-QUALITY pairs (the same
    positive-control battery that exposed Exact Match as blind, AUC 0.507).
    The judge is shown a real teacher-written distractor against a random
    one, the correct answer, or a perturbed near-miss. Correctness is known
    by construction, so accuracy is measurable directly.

    scripts/13_llm_judge.py refuses to score the experiment unless the judge
    clears a pre-set bar on this battery.

Bias controls
    - Judge model must differ in family from the generator (Qwen2.5-7B);
      self-preference is the most common failure of LLM-judge setups.
    - Pairwise, not absolute: 1-5 scales drift badly across calls.
    - A/B order randomised per item, and every pair scored in both orders so
      position bias is measurable rather than assumed away.
    - An explicit tie option: forcing a choice manufactures effect sizes.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


JUDGE_SYSTEM = (
    "You are an experienced mathematics teacher evaluating multiple-choice "
    "distractors (incorrect answer options). A good distractor is one a real "
    "student would plausibly choose because it follows from a specific, "
    "common misconception. A poor distractor is irrelevant, obviously wrong "
    "to any student, or is actually the correct answer. Judge impartially and "
    "concisely."
)

JUDGE_USER = """Question
Subject: {subject} | Construct: {construct}
{question}

Correct answer: {correct_answer}

Two candidate distractors:

[A] {option_a}

[B] {option_b}

Which is the better distractor for this question? Consider:
- Plausibility: would a real student actually choose it?
- Misconception grounding: does it follow from a specific, identifiable error?
- Educational usefulness: does choosing it reveal something about the student's thinking?
- Validity: it must be genuinely incorrect, not the correct answer.

Think briefly, then answer. Respond with JSON only:
{{"reasoning": "<one sentence>", "verdict": "A" | "B" | "TIE"}}"""


_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
_VERDICT = re.compile(r'"verdict"\s*:\s*"?(A|B|TIE)"?', re.IGNORECASE)


def parse_verdict(raw: str) -> Tuple[Optional[str], str]:
    """
    Extract the verdict from the judge's raw output.

    Returns (verdict, status) where verdict is 'A' | 'B' | 'TIE' | None and
    status is 'json', 'regex', or 'failed'. Unparseable responses are dropped
    from scoring rather than guessed at.
    """
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    m = _JSON_OBJ.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            v = str(obj.get("verdict", "")).strip().upper()
            if v in ("A", "B", "TIE"):
                return v, "json"
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
    m = _VERDICT.search(text)
    if m:
        return m.group(1).upper(), "regex"
    return None, "failed"


def build_judge_prompt(question: Dict, option_a: str, option_b: str) -> Tuple[str, str]:
    """Assemble the pairwise judging prompt."""
    return JUDGE_SYSTEM, JUDGE_USER.format(
        subject=question["subject"],
        construct=question["construct"],
        question=question["question_text"],
        correct_answer=question["correct_answer"],
        option_a=option_a,
        option_b=option_b,
    )


class LlamaJudge:
    """
    Local judge model, deliberately a different family from the generator.

    Default Llama-3.1-8B-Instruct in 4-bit; Qwen2.5-7B-Instruct produced the
    candidates, so judging with any Qwen variant would risk self-preference.
    Temperature defaults to 0 for reproducible verdicts.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        load_in_4bit: bool = True,
        max_new_tokens: int = 160,
        temperature: float = 0.0,
        seed: int = 42,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.seed = seed
        self._torch = torch

        print(f"  Loading judge: {model_name} (4bit={load_in_4bit})")
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

        # Some chat templates (notably Mistral-Instruct) do not support a
        # 'system' role and will raise or silently drop it. A dropped system
        # message leaves the judge with no instructions, which shows up as
        # severe A/B position bias rather than as an error. Detect it once
        # and fold the system text into the user turn when unsupported.
        self.supports_system = self._probe_system_role()
        print(f"  system role supported: {self.supports_system}"
              + ("" if self.supports_system else "  -> folding into user turn"))
        print("  Judge ready.")

    def _probe_system_role(self) -> bool:
        probe = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        try:
            out = self.tokenizer.apply_chat_template(
                probe, tokenize=False, add_generation_prompt=True)
        except Exception:
            return False
        return "S" in out            # False if the template silently dropped it

    def judge(self, system_prompt: str, user_prompt: str) -> str:
        from transformers import set_seed
        set_seed(self.seed)
        if self.supports_system:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}]
        else:
            messages = [{"role": "user",
                         "content": f"{system_prompt}\n\n{user_prompt}"}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:],
                                     skip_special_tokens=True)


def judge_pair_both_orders(
    judge, question: Dict, good: str, bad: str,
) -> Dict:
    """
    Score one pair in BOTH orders.

    Presenting (good, bad) and (bad, good) makes position bias measurable
    instead of merely randomised away:
        correct     - judge picked `good` in both orders (order-consistent)
        incorrect   - judge picked `bad` in both orders
        inconsistent- verdict flipped with position (pure position bias)
    """
    s1, u1 = build_judge_prompt(question, good, bad)   # good is A
    v1, st1 = parse_verdict(judge.judge(s1, u1))
    s2, u2 = build_judge_prompt(question, bad, good)   # good is B
    v2, st2 = parse_verdict(judge.judge(s2, u2))

    picks = []
    for v, good_is in ((v1, "A"), (v2, "B")):
        picks.append(None if v is None else ("good" if v == good_is
                                             else ("tie" if v == "TIE" else "bad")))
    if None in picks:
        outcome = "unparsed"
    elif picks[0] == picks[1]:
        outcome = {"good": "correct", "bad": "incorrect", "tie": "tie"}[picks[0]]
    else:
        outcome = "inconsistent"
    return {"verdict_order1": v1, "verdict_order2": v2,
            "pick_order1": picks[0], "pick_order2": picks[1],
            "outcome": outcome, "parse_status": f"{st1}/{st2}"}
