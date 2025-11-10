import os, sys, importlib, argparse
import random
import torch
from NeuralExec.utility import read_pickle
from NeuralExec.llm import load_llm
from NeuralExec.ex_triggers import NeuralExec
from NeuralExec.logger import Logger
from confs import hparams
from confs.evaluation_setup import vhparams
from NeuralExec.discrete_opt import WhiteBoxTokensOpt
from NeuralExec.utility import read_pickle, write_pickle
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def init_opt(wbo, conf_file, hparams):
    conf_file_name = conf_file.split('.')[-1]
    log_path = os.path.join(hparams['result_dir'], conf_file_name)

    if os.path.isfile(log_path):
        print(f"Resuming opt {log_path}")
        logger = read_pickle(log_path)
        hparams = logger.confs
        ne, _ = logger.get_last_adv_tok(best=True)
    else:
        print(f"Init opt {log_path}")
        # init/load log file
        logger = Logger(hparams)
        # init Neural Exec
        if 'boostrap_seed' in hparams:
            print("init_adv_seg boostrapping...")
            ne = wbo.init_adv_seg_boot(*hparams['boostrap_seed'],
                                       hparams['sep'])
        else:
            print("NexuralExec Random init...")
            ne = wbo.init_adv_seg(hparams['prefix_size'],
                                  hparams['postfix_size'],
                                  hparams['sep'])
    return ne, logger, log_path, hparams


def sample_batch(training_prompts, batch_size):
    return random.choices(training_prompts, k=batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='add targets for a LLM in the training, validation, and test set (the new LLM must be included in NeuralExec.llm first)')
    parser.add_argument('conf_file', type=str,
                       help='Path to the configuration file (e.g., confs.mistral)')
    parser.add_argument('--delimiter', type=str, default=None, choices=["TextTextText",
                                                          "TextTextTextMistral",
                                                          "SpclSpclSpcl"], help='delimiter to use')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch-size target generation')
    args = parser.parse_args()

    if args.delimiter is None:
        # load data
        training_prompts   = read_pickle(f'./data/prompts_training.pickle')
        validation_prompts = read_pickle(f'./data/prompts_validation.pickle')
    else:
        # load data
        training_prompts   = read_pickle(f'./data/prompts_training_{args.delimiter}.pickle')
        validation_prompts = read_pickle(f'./data/prompts_validation_{args.delimiter}.pickle')

    # load conf file
    conf = importlib.import_module(args.conf_file)
    hparams = conf.hparams

    # load LLM
    print(f"Loading LLM {hparams['llm']}...")
    # load llm
    llm = load_llm(hparams["llm"])

    # setup opt class
    wbo = WhiteBoxTokensOpt(llm, hparams)

    # init opt
    ne, logger, log_path, hparams = init_opt(wbo, args.conf_file, hparams)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    wbo.hparams = hparams

    # opt loop
    for i in range(hparams['number_of_rounds']):
        print(f'Start round {i + 1}/{hparams["number_of_rounds"]}')

        if i % hparams['eval_fq'] == 0: # fixme
            print("Starting evaluation...")
            eval_losses = wbo.eval_loss(validation_prompts, ne)
            print(eval_losses)
            logger.add_eval_log(ne, eval_losses, wbo.tokenizer)
            print("end evaluation.")

            logger.candidate_pool.insert_candidate(ne, eval_losses.mean())
            ne, best_loss_pool = logger.candidate_pool.get_best()
            print(eval_losses.mean(), best_loss_pool)

            write_pickle(log_path, logger)

        # sample batch for gradient
        train_batch = sample_batch(training_prompts, hparams['gradient_batch_size'])
        # compute gradient
        print("Computing gradient...")
        gradient, loss, losses = wbo.get_gradient_accum(ne, train_batch)
        logger.add_train_log(loss, ne, wbo.tokenizer)

        # sample candidate solutions
        new_candidate_tok = wbo.sample_new_candidates(ne, gradient)
        # filter out bad ones
        new_candidate_tok = wbo.filter_candidates(ne, new_candidate_tok)
        # pick new solution
        ne, best_candidate_loss, _, _ = wbo.test_candidates(train_batch, new_candidate_tok)
