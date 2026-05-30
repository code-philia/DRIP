from __future__ import annotations
import os
import argparse
import json
import re
from typing import Dict, Optional

from tqdm import tqdm

from testing.test import load_full_model, test_model_output
from config import PROMPT_FORMAT, DEFAULT_SYSTEM_PROMPT
from testing.tap.utils import _inject, _generate
from testing.pismith.utils import (
    extract_injected_task,
    format_probe_in_task,
    make_witness_judge,
)
from testing.pismith.trainer import PISmithInference


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name_or_path", type=str, nargs="+", required=True)
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--customized_model_class", type=str, default="")
    parser.add_argument("--attack_model_path", type=str, default="./pismith_ckpt/sep/")
    parser.add_argument("--attack_base_model", type=str, default=None,
                        help="Base model for LoRA. If None, read from adapter_config.json.")
    parser.add_argument("--data_path", type=str, default="./datasets/SEP_dataset.json")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--inject_position", type=str, default="end",
                        choices=["end", "start", "random"])
    parser.add_argument("--target_max_new_tokens", type=int, default=512)
    parser.add_argument("--attack_temperature", type=float, default=1.0)
    parser.add_argument("--attack_max_new_tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=200,
                        help="Max number of test samples to evaluate.")
    args = parser.parse_args()
    args.model_name_or_path = args.model_name_or_path[0]

    # ── Load target model ─────────────────────────────────────────────────────
    model, tokenizer, frontend_delimiters, _ = load_full_model(
        args.model_name_or_path,
        customized_model_class=args.customized_model_class,
        base_model_path=args.base_model_path,
    )
    fmt = PROMPT_FORMAT[frontend_delimiters]

    # ── Target query function (closure over witness for sanitization) ─────────
    def make_target_query_fn(context: str, target_inst: str, witness: str):
        _witness_re = re.compile(re.escape(witness), re.IGNORECASE) if witness else None

        def target_query_fn(_inst: str, _ctx: str, attack_prompt: str) -> str:
            sanitized = (_witness_re.sub("", attack_prompt).strip()
                         if _witness_re else attack_prompt)
            injected_context = _inject(context, sanitized, position=args.inject_position)
            prompt = fmt["prompt_input"].format_map({
                "system": DEFAULT_SYSTEM_PROMPT,
                "instruction": target_inst,
                "input": injected_context,
            })
            return _generate(prompt, model, tokenizer, frontend_delimiters,
                             max_new_tokens=args.target_max_new_tokens,
                             customized_model_class=args.customized_model_class)
        return target_query_fn

    # ── Output path ───────────────────────────────────────────────────────────
    model_path = args.model_name_or_path
    log_path = f"{model_path}-log" if not os.path.exists(model_path) else model_path
    os.makedirs(log_path, exist_ok=True)
    out_path = os.path.join(log_path, "predictions_on_sep_pismith.jsonl")

    output: list = []
    start_idx = 0
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            try:
                output = json.load(f)
                start_idx = len(output)
                print(f"[INFO] Resuming from index {start_idx}.")
            except json.JSONDecodeError:
                print("[WARNING] Output file corrupted, starting from scratch.")
    output = output[:args.limit]

    # ── Load attack LLM (base + LoRA) via PISmithInference ────────────────────
    # Placeholder fns; we'll override per-sample below
    attack_engine = PISmithInference(
        attack_model_path=args.attack_model_path,
        base_model_name=args.attack_base_model,
        target_query_fn=lambda *_: "",   # placeholder, set per-sample
        judge_fn=lambda _: False,        # placeholder, set per-sample
        n_samples=args.n_samples,
        temperature=args.attack_temperature,
        max_new_tokens=args.attack_max_new_tokens,
        witness_filter=True,
    )

    # ── Load dataset ──────────────────────────────────────────────────────────
    with open(args.data_path, "r") as f:
        dataset = json.load(f)

    # ── Main eval loop ────────────────────────────────────────────────────────
    for i, sample in enumerate(tqdm(dataset, desc="SEP PISmith")):
        if i < start_idx or i >= args.limit:
            continue

        witness       = sample["witness"]
        target_inst   = sample["system_prompt_clean"]
        context       = sample["prompt_clean"]
        injected_task = extract_injected_task(sample)

        # Wire per-sample target + judge into the inference engine
        attack_engine.target_query_fn = make_target_query_fn(context, target_inst, witness)
        attack_engine.judge_fn        = make_witness_judge(witness)

        # output2: probe-in-task reference (baseline: probe embedded in the task itself)
        probe_in_task_prompt = format_probe_in_task(sample, fmt)
        _, _, _, ref_out = test_model_output(
            [probe_in_task_prompt], model, tokenizer,
            attack_log_file=None, print_results=False,
            frontend_delimiters=frontend_delimiters,
        )
        output2: str = ref_out[0][0]

        # Generate N attack rollouts and query target
        result = attack_engine.attack_one(
            target_inst=target_inst,
            context=context,
            injected_task=injected_task,
            witness=witness,
        )

        output.append({
            "output1_probe_in_data": result["best_response"],
            "output2_probe_in_task": output2,
            "pismith_asr_at_n":      result["asr_at_n"],
            "pismith_asr_at_1":      result["asr_at_1"],
            "pismith_best_prompt":   result["best_prompt"],
            "model":                 args.customized_model_class or args.model_name_or_path,
            "instructions": {
                "input_1": fmt["prompt_input"].format_map({
                    "system": DEFAULT_SYSTEM_PROMPT,
                    "instruction": target_inst,
                    "input": _inject(context, result["best_prompt"], position=args.inject_position),
                }),
                "input_2": probe_in_task_prompt,
            },
            "data":     sample,
            "rollouts": result["rollouts"],
        })

        if i % 5 == 0 or i == len(dataset) - 1:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            n = len(output)
            asr_n = sum(1 for r in output if r["pismith_asr_at_n"]) / n
            print(f"  [saved {n}] ASR@{args.n_samples}={asr_n:.3f}")

    n = len(output)
    asr_n = sum(1 for r in output if r["pismith_asr_at_n"]) / n
    asr_1 = sum(1 for r in output if r["pismith_asr_at_1"]) / n
    print(f"\nDone. n={n}")
    print(f"  ASR@{args.n_samples}: {asr_n:.3f}")
    print(f"  ASR@1: {asr_1:.3f}")


if __name__ == "__main__":
    main()