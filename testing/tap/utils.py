from __future__ import annotations

import os
import random
import threading
import torch
from typing import Callable, Dict, List, Optional, Tuple

from openai import OpenAI
from data_generation.sft_data_loader import _tokenize_fn
from dataclasses import dataclass
import json
import re

_generate_lock = threading.Lock()

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _inject(context: str, attack_prompt: str, position: str = "end") -> str:
    if position == "end":
        sep = "\n" if context and not context.endswith("\n") else ""
        return context + sep + attack_prompt
    if position == "start":
        sep = "\n" if context else ""
        return attack_prompt + sep + context
    if position == "random":
        words = context.split()
        idx = random.randint(0, len(words))
        words.insert(idx, attack_prompt)
        return " ".join(words)
    raise ValueError(f"Unknown inject_position: {position!r}")


def _openai_chat(
        client: OpenAI,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 1.0,
        max_retries: int = 5,
        base_delay: float = 2.0,
) -> str:
    import time
    from openai import RateLimitError, APIConnectionError, APITimeoutError, InternalServerError

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)  # exponential backoff: 2, 4, 8, 16, 32s
            print(f"[openai] {type(e).__name__} on attempt {attempt + 1}, retrying in {delay:.0f}s...")
            time.sleep(delay)
    return ""  # unreachable


@torch.inference_mode()
def _generate(
    prompt: str,
    model,
    tokenizer,
    frontend_delimiters: str,
    max_new_tokens: int = 512,
    customized_model_class: str = "",
) -> str:
    """Mirrors test_model_output: tokenize → greedy generate → decode.

    If customized_model_class is set (DRIP model), computes expert_labels
    via compute_gate=True and passes them to model.generate().
    For standard HF models (customized_model_class=""), expert_labels is skipped.
    """
    with _generate_lock:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

        use_expert_labels = bool(customized_model_class)
        toks = _tokenize_fn(
            [prompt], tokenizer,
            frontend_delimiters=frontend_delimiters,
            compute_gate=use_expert_labels,
        )
        inp = toks["input_ids"][0].unsqueeze(0).to(model.device)

        gen_kwargs = dict(
            attention_mask=torch.ones_like(inp),
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,
            top_p=None,  # suppress "not valid" warning
        )
        if use_expert_labels:
            gen_kwargs["expert_labels"] = toks["expert_labels"][0].unsqueeze(0).to(model.device)

        out_ids  = model.generate(inp, **gen_kwargs)
        response = tokenizer.decode(out_ids[0, inp.shape[1]:], skip_special_tokens=True)
        torch.cuda.empty_cache()
    return response


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def build_tap_callables(
    *,
    model,
    tokenizer,
    fmt: Dict[str, str],
    frontend_delimiters: str,
    target_inst: str,
    context: str,
    witness: str,
    inject_position: str = "end",
    target_max_new_tokens: int = 512,
    customized_model_class: str = "",
    openai_api_key: Optional[str] = None,
    attacker_model: str = "gpt-4o",
    evaluator_model: str = "gpt-4o-mini",
    attacker_max_tokens: int = 1024,
    attacker_temperature: float = 1.0,
    evaluator_max_tokens: int = 128,
    evaluator_temperature: float = 0.0,
) -> Tuple[Callable, Callable, Callable]:
    """
    Returns (attacker_chat, evaluator_chat, target_query).

    target_query strips the witness word from P before injection.
    If customized_model_class is set (DRIP model), expert_labels are computed
    and passed to model.generate(). For standard HF models, expert_labels is skipped.
    """
    assert "prompt_input_tool" in fmt, (
        "fmt must contain 'prompt_input_tool'. "
        "Build it from the 4-delimiter template before calling this function."
    )

    client = OpenAI(api_key=openai_api_key or os.environ["OPENAI_API_KEY"])
    _witness_re = re.compile(re.escape(witness), re.IGNORECASE) if witness else None

    def attacker_chat(messages: List[Dict[str, str]]) -> str:
        return _openai_chat(client, attacker_model, messages,
                            max_tokens=attacker_max_tokens,
                            temperature=attacker_temperature)

    def evaluator_chat(messages: List[Dict[str, str]]) -> str:
        return _openai_chat(client, evaluator_model, messages,
                            max_tokens=evaluator_max_tokens,
                            temperature=evaluator_temperature)

    def target_query(attack_prompt: str) -> str:
        sanitized = _witness_re.sub("", attack_prompt).strip() if _witness_re else attack_prompt
        injected_context = _inject(context, sanitized, position=inject_position)
        prompt = fmt["prompt_input_tool"].format_map({
            "instruction": target_inst,
            "input":       injected_context,
        })
        return _generate(prompt, model, tokenizer, frontend_delimiters,
                         max_new_tokens=target_max_new_tokens,
                         customized_model_class=customized_model_class)

    return attacker_chat, evaluator_chat, target_query


@dataclass
class AttackerFeedback:
    """Parsed attacker LLM output: improvement explanation + new attack prompt."""
    improvement: str
    prompt: str


def parse_attacker_feedback(content: str) -> Optional[AttackerFeedback]:
    """
    Parse attacker LLM response to extract {"improvement": ..., "prompt": ...} JSON.

    Handles markdown code blocks and malformed JSON by trying progressively
    shorter suffixes from the first '{' to the last valid '}'.

    Handles markdown code blocks and malformed JSON by trying progressively
    shorter suffixes from the first '{' to the last valid '}'.
    """
    if not content:
        return None

    content = content.strip()

    # Strip markdown code block if present
    if "```" in content:
        m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content)
        if m:
            content = m.group(1)

    # Find first '{'
    start = content.find("{")
    if start == -1:
        return None

    # Try progressively shorter suffixes to get valid JSON
    rest = content[start:]
    for end in range(len(rest), 0, -1):
        if rest[end - 1] != "}":
            continue
        try:
            obj = json.loads(rest[:end])
            if isinstance(obj, dict) and ("improvement" in obj or "prompt" in obj):
                return AttackerFeedback(
                    improvement=obj.get("improvement", ""),
                    prompt=obj.get("prompt", ""),
                )
        except json.JSONDecodeError:
            continue
    return None