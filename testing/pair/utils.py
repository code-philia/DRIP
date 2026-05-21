"""
pair/utils.py

Shared utilities for PAIR attack.
build_pair_callables() is identical to build_tap_callables() but only returns
(attacker_chat, target_query) — PAIR has no evaluator.
_inject, _generate, _openai_chat, AttackerFeedback, parse_attacker_feedback
are re-exported from testing.tap.utils to avoid duplication.
"""

from __future__ import annotations

import os
import re
from typing import Callable, Dict, List, Optional, Tuple

from openai import OpenAI

# Re-use all low-level helpers from tap
from testing.tap.utils import (
    _inject,
    _generate,
    _openai_chat,
    AttackerFeedback,
    parse_attacker_feedback,
)


def build_pair_callables(
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
    attacker_max_tokens: int = 1024,
    attacker_temperature: float = 1.0,
) -> Tuple[Callable, Callable]:
    """
    Returns (attacker_chat, target_query).

    PAIR does not use an evaluator — no on-topic check, no scorer.
    target_query strips the witness word from P (same constraint as TAP).
    """
    assert "prompt_input_tool" in fmt

    client = OpenAI(api_key=openai_api_key or os.environ["OPENAI_API_KEY"])
    _witness_re = re.compile(re.escape(witness), re.IGNORECASE) if witness else None

    def attacker_chat(messages: List[Dict[str, str]]) -> str:
        return _openai_chat(client, attacker_model, messages,
                            max_tokens=attacker_max_tokens,
                            temperature=attacker_temperature)

    def target_query(attack_prompt: str) -> str:
        sanitized = _witness_re.sub("", attack_prompt).strip() if _witness_re else attack_prompt
        injected_context = _inject(context, sanitized, position=inject_position)
        prompt = fmt["prompt_input_tool"].format_map({
            "instruction": target_inst,
            "input":       injected_context,
        })
        return _generate(prompt, model, tokenizer, frontend_delimiters,
                         max_new_tokens=target_max_new_tokens,
                         customized_model_class=customized_model_class
                         )

    return attacker_chat, target_query