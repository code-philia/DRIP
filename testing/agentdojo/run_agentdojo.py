#!/usr/bin/env python3
"""
Run AgentDojo benchmarks for LlamaForCausalLMFuse (or any fuse-model variant).

Basic usage
-----------
# Utility only (no attack), banking suite:
python run_agentdojo.py \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct-TextTextText-instfuse-sep-none-newdata-dpo \
    --model-class LlamaForCausalLMFuse \
    --suites banking

# Under important_instructions attack, all suites:
python run_agentdojo.py \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct-TextTextText-instfuse-sep-none-newdata-dpo \
    --model-class LlamaForCausalLMFuse \
    --attack important_instructions \
    --suites banking slack travel workspace

# Compare ours vs SecAlign baseline (no custom class):
python run_agentdojo.py \
    --model-path meta-llama/Meta-SecAlign-8B-merged \
    --model-class "" \
    --attack important_instructions \
    --suites banking
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

# ── Project imports ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fuse_llm import build_fuse_pipeline

from agentdojo.attacks.attack_registry import ATTACKS, load_attack
from agentdojo.benchmark import (
    SuiteResults,
    TaskResults,
    benchmark_suite_without_injections,
)
from agentdojo.logging import OutputLogger
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.task_suite.task_suite import TaskSuite
from agentdojo.base_tasks import BaseUserTask, BaseInjectionTask
from agentdojo.functions_runtime import FunctionCall  # noqa: F401

TaskResults.model_rebuild()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ALL_SUITES = ["banking", "slack", "travel", "workspace"]


# ════════════════════════════════════════════════════════════════════════════
# Manual injection benchmark
# ════════════════════════════════════════════════════════════════════════════

def benchmark_suite_manual_injections(
    pipeline,
    suite: TaskSuite,
    attacker,
    user_tasks: list[str] | None = None,
    injection_tasks: list[str] | None = None,
    verbose: bool = True,
) -> SuiteResults:
    """
    Drop-in replacement for benchmark_suite_with_injections that manually
    builds injections and calls suite.run_task_with_pipeline directly.

    This bypasses benchmark_suite_with_injections entirely so we can verify
    exactly what injection strings reach the environment.
    """
    utility_results:   dict[tuple[str, str], bool] = {}
    security_results:  dict[tuple[str, str], bool] = {}
    inj_utility_results: dict[str, bool] = {}

    # Filter tasks if requested
    all_user_tasks = suite.user_tasks
    all_inj_tasks  = suite.injection_tasks

    selected_user_tasks = (
        {k: v for k, v in all_user_tasks.items() if k in user_tasks}
        if user_tasks else all_user_tasks
    )
    selected_inj_tasks = (
        {k: v for k, v in all_inj_tasks.items() if k in injection_tasks}
        if injection_tasks else all_inj_tasks
    )

    # Check injection task utility once with ground-truth pipeline
    # (mirrors what benchmark_suite_with_injections does internally)
    from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
    for inj_task_id, inj_task in selected_inj_tasks.items():
        gt_pipeline = GroundTruthPipeline(inj_task)
        _, inj_util = suite.run_task_with_pipeline(
            agent_pipeline=gt_pipeline,
            user_task=next(iter(selected_user_tasks.values())),
            injection_task=inj_task,
            injections={},
        )
        inj_utility_results[inj_task_id] = inj_util

    n_total = len(selected_user_tasks) * len(selected_inj_tasks)
    n_done  = 0

    for user_task_id, user_task in selected_user_tasks.items():
        for inj_task_id, inj_task in selected_inj_tasks.items():
            n_done += 1

            # ── Build injection strings ──────────────────────────────────
            injections = attacker.attack(user_task, inj_task)

            # ── Diagnostic: verify injections survive into the environment
            env_check = suite.load_and_inject_default_environment(injections)
            env_str   = str(env_check.model_dump())
            for vec_key, vec_val in injections.items():
                snippet = vec_val[:60].replace('\n', '\\n')
                found   = vec_val[:40] in env_str
                logger.debug(
                    "  env-check [%s]: in_env=%s  snippet=%s",
                    vec_key, found, repr(snippet),
                )
                if not found:
                    logger.warning(
                        "INJECTION NOT IN ENV — vector '%s' did not survive "
                        "load_and_inject_default_environment. "
                        "Snippet: %s", vec_key, repr(snippet),
                    )

            # ── Run the task ─────────────────────────────────────────────
            if verbose:
                logger.info(
                    "[%d/%d] %s × %s | vectors: %s",
                    n_done, n_total, user_task_id, inj_task_id, list(injections.keys()),
                )

            utility, security = suite.run_task_with_pipeline(
                agent_pipeline=pipeline,
                user_task=user_task,
                injection_task=inj_task,
                injections=injections,
            )

            utility_results[(user_task_id, inj_task_id)]  = utility
            security_results[(user_task_id, inj_task_id)] = security

            if verbose:
                logger.info(
                    "  → utility=%s  security(injection_succeeded)=%s",
                    utility, security,
                )

    return SuiteResults(
        utility_results=utility_results,
        security_results=security_results,
        injection_tasks_utility_results=inj_utility_results,
    )


# ════════════════════════════════════════════════════════════════════════════
# Results display
# ════════════════════════════════════════════════════════════════════════════

def show_results(suite_name: str, results: SuiteResults, with_attack: bool) -> None:
    util_vals = list(results["utility_results"].values())
    avg_util  = sum(util_vals) / len(util_vals) if util_vals else 0.0
    print(f"\n{'=' * 55}")
    print(f"Suite: {suite_name}")
    print(f"  Utility  : {avg_util * 100:.1f}%  ({sum(util_vals)}/{len(util_vals)})")

    if with_attack:
        sec_vals = list(results["security_results"].values())
        avg_sec  = sum(sec_vals) / len(sec_vals) if sec_vals else 0.0
        # security=True means injection did NOT succeed (model was secure)
        print(f"  Security : {avg_sec * 100:.1f}%  ({sum(sec_vals)}/{len(sec_vals)})")

        inj_vals = results.get("injection_tasks_utility_results", {})
        if inj_vals:
            n_pass = sum(inj_vals.values())
            print(f"  Injection task utility: {n_pass}/{len(inj_vals)}")
    print(f"{'=' * 55}")


def save_summary(
    all_results: dict[str, SuiteResults],
    model_name:  str,
    attack:      str | None,
    output_path: Path,
) -> None:
    summary = {}
    for suite_name, results in all_results.items():
        util_vals = list(results["utility_results"].values())
        sec_vals  = list(results["security_results"].values())
        summary[suite_name] = {
            "utility":  sum(util_vals) / len(util_vals) if util_vals else None,
            "security": sum(sec_vals)  / len(sec_vals)  if sec_vals  else None,
            "n_pairs":  len(util_vals),
        }
    out = {"model": model_name, "attack": attack, "results": summary}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    logger.info("Summary saved to %s", output_path)


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark LlamaForCausalLMFuse (or variants) on AgentDojo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-m', '--model_name_or_path', type=str, nargs="+")
    p.add_argument('--customized_model_class', type=str, default='')
    p.add_argument("--model-name", default=None)
    p.add_argument("--no-expert-labels", action="store_true")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-iters", type=int, default=15)
    p.add_argument("--benchmark-version", default="v1.2.1")
    p.add_argument("--suites", nargs="+", default=ALL_SUITES,
                   choices=ALL_SUITES + ["all"])
    p.add_argument("--attack", default=None,
                   choices=list(ATTACKS) + [None])   # type: ignore[arg-type]
    p.add_argument("--user-tasks", nargs="*", default=None)
    p.add_argument("--injection-tasks", nargs="*", default=None)
    p.add_argument("--logdir", type=Path, default=Path("./agentdojo_runs"))
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--system-message", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.model_name_or_path = args.model_name_or_path[0]

    import torch
    torch_dtype = {"bfloat16": torch.bfloat16,
                   "float16":  torch.float16,
                   "float32":  torch.float32}[args.dtype]

    suites = ALL_SUITES if "all" in args.suites else args.suites

    logger.info("Loading model from %s ...", args.model_name_or_path)
    pipeline = build_fuse_pipeline(
        model_path=args.model_name_or_path,
        customized_model_class=args.customized_model_class or None,
        model_name=f"local/{args.model_name_or_path}",
        system_message=args.system_message,
        torch_dtype=torch_dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        pass_expert_labels=not args.no_expert_labels,
        max_iters=args.max_iters,
    )
    pipeline.name = f"local/{args.model_name_or_path}"
    logger.info("Pipeline ready: %s", pipeline.name)

    all_results: dict[str, SuiteResults] = {}

    with OutputLogger(str(args.logdir)):
        for suite_name in suites:
            logger.info("Running suite: %s", suite_name)
            suite = get_suite(args.benchmark_version, suite_name)

            if args.attack is None:
                results = benchmark_suite_without_injections(
                    agent_pipeline=pipeline,
                    suite=suite,
                    logdir=args.logdir,
                    force_rerun=args.force_rerun,
                    user_tasks=args.user_tasks or None,
                    benchmark_version=args.benchmark_version,
                )
            else:
                attacker = load_attack(args.attack, suite, pipeline)

                # Add this right after attacker = load_attack(...) in main()
                first_ut = list(suite.user_tasks.values())[0]
                first_it = list(suite.injection_tasks.values())[0]
                injections = attacker.attack(first_ut, first_it)

                env = suite.load_and_inject_default_environment(injections)
                env_dict = env.model_dump()

                # Print the full environment so you can see where injection vectors live
                import json
                print(json.dumps(env_dict, indent=2, default=str))
                # ── Manual injection loop (replaces benchmark_suite_with_injections)
                results = benchmark_suite_manual_injections(
                    pipeline=pipeline,
                    suite=suite,
                    attacker=attacker,
                    user_tasks=args.user_tasks,
                    injection_tasks=args.injection_tasks,
                    verbose=True,
                )

            all_results[suite_name] = results
            show_results(suite_name, results, with_attack=(args.attack is not None))

    if len(suites) > 1:
        all_util = [v for r in all_results.values() for v in r["utility_results"].values()]
        all_sec  = [v for r in all_results.values() for v in r["security_results"].values()]
        print(f"\n{'=' * 55}")
        print(f"OVERALL  ({len(suites)} suites)")
        print(f"  Utility  : {sum(all_util)/len(all_util)*100:.1f}%")
        if args.attack:
            print(f"  Security : {sum(all_sec)/len(all_sec)*100:.1f}%")
        print(f"{'=' * 55}")

    summary_path = args.logdir / f"{pipeline.name}_{args.attack or 'no-attack'}_summary.json"
    save_summary(all_results, pipeline.name, args.attack, summary_path)


if __name__ == "__main__":
    main()