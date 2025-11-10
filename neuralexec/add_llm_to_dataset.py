import os, argparse
import math, tqdm
from copy import deepcopy

from NeuralExec.llm import load_llm
from NeuralExec.utility import read_pickle, write_pickle
from NeuralExec.adv_prompts import Prompt

from confs import hparams
from confs.evaluation_setup import vhparams


def get_instructions(dbs):
    insts = {}
    for db in dbs:
        for p in db:
            insts[p.payload] = p.target
    return insts


def put_instructions_back(dbs, insts, delimiter):
    for db in dbs:
        for p in db:
            key = p.payload
            p.target = insts[key]
            p.delimiter = delimiter # update delimiter


def make_targets(llm, insts, batch_size, vhparams, delimiter, verbose=0):
    num_batches = math.ceil(len(insts) / batch_size)

    insts = deepcopy(insts)

    keys = list(insts.keys())
    for i in tqdm.trange(num_batches):
        batch_keys = keys[batch_size * i:batch_size * (i + 1)]

        prompts = [Prompt(key, delimiter)(llm.tokenizer) for key in batch_keys]
        targets = llm.generate(
            prompts,
            max_new_tokens=vhparams['max_new_tokens'],
            do_sample=False
        )

        for key, target in zip(batch_keys, targets):
            if verbose > 0:
                print(f'[{key}] --> [{target}]')
            insts[key][llm.llm_name] = target

        # break # fixme

    return insts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='add targets for a LLM in the training, validation, and test set (the new LLM must be included in NeuralExec.llm first)')
    parser.add_argument('--llm_name', type=str, help='huggingface llm path e.g., "meta-llama/Meta-Llama-3-8B-Instruct" ')
    parser.add_argument('--delimiter', type=str, choices=["TextTextText",
                                                          "TextTextTextMistral",
                                                          "SpclSpclSpcl"], help='delimiter to use')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch-size target generation')
    args = parser.parse_args()

    # load llm
    llm = load_llm(args.llm_name)

    # load instruction from existing datasets
    db_paths = (*hparams['dataset_paths'], vhparams['testset_path'])
    out_paths = [db.replace(".pickle", f"_{args.delimiter}.pickle") for db in db_paths]
    if os.path.exists(out_paths[0]):
        dbs   = [read_pickle(path) for path in out_paths] # resume from last time
        print("Resume from", out_paths)
    else:
        dbs   = [read_pickle(path) for path in db_paths]
        print("Load original", db_paths)
    insts = get_instructions(dbs)

    # compute targets
    print("Computing targets...")
    new_insts = make_targets(llm, insts, args.batch_size, vhparams, args.delimiter, 0)
    #
    # # save back
    put_instructions_back(dbs, new_insts, args.delimiter)
    for path, db in zip(out_paths, dbs):
        print(f'Saving {path}')
        write_pickle(path, db)

    # conda activate gcg
    # cd neuralexec/
    # CUDA_VISIBLE_DEVICES=0 python add_llm_to_dataset.py --llm_name /../PromptInjection/meta-llama/Meta-Llama-3-8B-Instruct-TextTextText-instfuse-sep-none-newdata-dpo --delimiter TextTextText
    # ise
    # possep
    # CUDA_VISIBLE_DEVICES=4 python add_llm_to_dataset.py --llm_name /../PromptInjection/meta-llama/Meta-Llama-3-8B-Instruct-SpclSpclSpcl-struq-sep-none --delimiter SpclSpclSpcl
    # CUDA_VISIBLE_DEVICES=4 python add_llm_to_dataset.py --llm_name /../PromptInjection/meta-llama/Meta-Llama-3-8B-Instruct-SpclSpclSpcl-secalign-sep-none --delimiter SpclSpclSpcl