import fastchat
from ml_collections import config_dict
from gcg.gcg import GCGConfig, run
from config import (
    DELIMITERS,
    PROMPT_FORMAT,
    SYS_INPUT,
    TEST_INJECTED_PROMPT,
    TEST_INJECTED_WORD,
)
import dataclasses
import torch
from data_generation.struq import jload, _tokenize_fn
from testing.test import load_full_model, test_model_output, recursive_filter
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy
from tqdm import tqdm
import csv
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Testing a model with a specific attack')
    parser.add_argument('-m', '--model_name_or_path', type=str, nargs="+")
    parser.add_argument('-a', '--attack', type=str, default='gcg',
                        choices=["gcg"],
                        nargs='+')
    parser.add_argument('-d', '--defense', type=str, default='none',
                        choices=['none', 'sandwich', 'instructional', 'reminder', 'isolation', 'incontext'],
                        help='Baseline test-time zero-shot prompting defense')
    parser.add_argument('--data_path', type=str, default='datasets/davinci_003_outputs.json')
    parser.add_argument('--pass_expert_labels', default=False,
                        help="Whether to past expert labels instruction/data as an input", action='store_true')
    parser.add_argument('--customized_model_class', type=str, help="Customized model class", default='')
    args = parser.parse_args()
    args.model_name_or_path = args.model_name_or_path[0]

    model, tokenizer, frontend_delimiters, training_attacks = load_full_model(args.model_name_or_path, customized_model_class=args.customized_model_class)

    prompt_format = PROMPT_FORMAT[frontend_delimiters]

    config = GCGConfig(
        num_steps=250,
        search_width=512,
        topk=256,
        batch_size=120,
        seed=42,
        verbosity="WARNING",
        use_prefix_cache=False if args.pass_expert_labels else True
    )

    response_delimiter = DELIMITERS[frontend_delimiters][2]
    data = [d for d in jload(args.data_path) if d["input"] != ""]
    model_path = args.model_name_or_path
    a = "gcg"
    log_path = f"{model_path}-log" if not os.path.exists(model_path) else model_path
    attack_log_file = os.path.join(log_path, f"{a}-{args.defense}-{TEST_INJECTED_WORD}.csv")
    summary_path = os.path.join(log_path, "summary.tsv")

    llm_inputs = []
    for i, d in tqdm(enumerate(data), "Attacking samples..."):
        d_item = deepcopy(d)
        # gcg attack
        d_item["input"] += " " + TEST_INJECTED_PROMPT.capitalize()
        orig_str = prompt_format['prompt_input'].format_map(d_item)

        result = run(model=model,
                     tokenizer=tokenizer,
                     messages=orig_str,
                     target=TEST_INJECTED_WORD.lower(),
                     frontend_delimiter=frontend_delimiters,
                     config=config,
                     pass_expert_labels=args.pass_expert_labels)

        d_item["input"] = d_item["input"] + " " + result.best_string
        attacked_str    = prompt_format['prompt_input'].format_map(d_item)
        llm_inputs.append(attacked_str)

    open(attack_log_file, 'w').close()
    in_response, begin_with, outputs = test_model_output(llm_input=llm_inputs,
                                                         model=model,
                                                         tokenizer=tokenizer,
                                                         attack_log_file=attack_log_file,
                                                         frontend_delimiters=frontend_delimiters,
                                                         pass_expert_labels=args.pass_expert_labels,
                                                         print_results=True)
    #
    print(
        f"\n{a} success rate {in_response} / {begin_with} (in-response / begin_with) on {args.model_name_or_path}, "
        f"delimiters {frontend_delimiters}, "
        f"training-attacks {training_attacks},"
        f"zero-shot defense {args.defense}\n"
    )
    with open(summary_path, "a") as outfile:
        outfile.write(f"{a}\t{in_response}\t{begin_with}\t{args.defense}_{TEST_INJECTED_WORD}\n")







