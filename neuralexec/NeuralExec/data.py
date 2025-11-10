# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import re
from copy import deepcopy
import io, json
from .config import PROMPT_FORMAT, OTHER_DELM_FOR_TEST, OTHER_DELM_TOKENS, \
    IGNORE_INDEX, DELIMITERS, SPECIAL_DELM_TOKENS, TEXTUAL_DELM_TOKENS
import torch
import transformers
from typing import Dict

def jload(f, mode="r"):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    jdict = json.load(f)
    f.close()
    return jdict


def jdump(obj, f, mode="w", indent=4, default=str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def format_with_other_delimiters(text, test=False):
    test_idx = - OTHER_DELM_FOR_TEST
    mark = np.random.choice(
        OTHER_DELM_TOKENS['mark'][test_idx:] if test else OTHER_DELM_TOKENS['mark'][:test_idx]) + ':'

    def sample_delm(delm_name):
        role_name = 'user' if (delm_name == 'inst' or delm_name == 'inpt') else 'asst'
        if test:
            role = np.random.choice(OTHER_DELM_TOKENS[role_name][test_idx:])
            delm = np.random.choice(OTHER_DELM_TOKENS[delm_name][test_idx:])
        else:
            role = np.random.choice(OTHER_DELM_TOKENS[role_name][:test_idx])
            delm = np.random.choice(OTHER_DELM_TOKENS[delm_name][:test_idx])

        p = np.random.rand()
        if p < 1 / 3:
            return (role + delm).upper()
        elif p < 2 / 3:
            return (role + delm).lower()
        else:
            return role + delm

    for delm in DELIMITERS.values():
        if '' in delm or ' ' in delm: continue
        text = text.replace(delm[0], mark.format(s=sample_delm('inst')))
        text = text.replace(delm[1], mark.format(s=sample_delm('inpt')))
        text = text.replace(delm[2], mark.format(s=sample_delm('resp')))
    return text


def generate_training_data(data_dicts, prompt_dict_name, attack, frontend_delimiters, tokenizer):
    prompt_dict = PROMPT_FORMAT[prompt_dict_name]

    if attack == 'None':
        return [prompt_dict["prompt_input"].format_map(example) for example in data_dicts], \
               [f"{example['output']}{tokenizer.eos_token}" for example in data_dicts]

    if attack == 'Completion':
        ref_inst_resp = {}
        for ref_sample in jload('./datasets/alpaca_data.json'):
            ref_inst_resp[ref_sample['instruction']] = ref_sample['output']
    sources = []

    for i in range(len(data_dicts)):
        # no input => no attack
        if data_dicts[i].get("input", "") == "":
            sources.append(prompt_dict["prompt_input"].format_map(data_dicts[i]))
        else:
            # randomly choose another instruction from the same dataset
            injected_sample = deepcopy(np.random.choice(data_dicts))
            injected_sample['instruction'] = injected_sample['instruction']
            if injected_sample['instruction'][-1] == '?':
                injected_prompt = 'answer the following question. ' + \
                                  injected_sample['instruction'] + ' ' + \
                                  injected_sample['input']
            else:
                injected_prompt = injected_sample['instruction'][0].lower() + \
                                  injected_sample['instruction'][1:] + ' ' + \
                                  injected_sample['input']

            data_dicts_item = deepcopy(data_dicts[i])
            if data_dicts_item['input'][-1] != '.': data_dicts_item['input'] += '.'
            if attack == 'Naive':
                data_dicts_item['input'] += ' ' + injected_prompt[0].upper() + injected_prompt[1:]
            elif attack == 'Ignore':
                data_dicts_item['input'] += ' ' + np.random.choice(
                    IGNORE_ATTACK_SENTENCES['train']) + ' ' + injected_prompt
            elif attack == 'Completion':
                data_dicts_item['input'] += '\n\n' + \
                                            DELIMITERS[frontend_delimiters][2] \
                                            + '\n' + \
                                            ref_inst_resp.get(data_dicts_item['instruction'],
                                                              data_dicts_item['output']) + \
                                            '\n\n' + \
                                            DELIMITERS[frontend_delimiters][0] + \
                                            '\n' + \
                                            injected_prompt.capitalize()
                if injected_sample['input'] != '':
                    data_dicts_item['input'] += '\n\n' + \
                                                DELIMITERS[frontend_delimiters][1] + \
                                                '\n' + \
                                                injected_sample['input']
                data_dicts_item['input'] = format_with_other_delimiters(data_dicts_item['input'], test=False)

            else:
                raise NotImplementedError

            sources.append(prompt_dict["prompt_input"].format_map(data_dicts_item))

    return sources, [f"{example['output']}{tokenizer.eos_token}" for example in data_dicts]


def find_first_occurrence(seq, separator):
    for i in range(len(seq) - len(separator) + 1):
        if torch.equal(seq[i:i + len(separator)], separator):  # Check if slice matches the separator
            return i  # Return the ENDING index of the first match
    return -1  # Return -1 if separator is not found


def find_last_occurrence(seq, separator):
    last_index = -1
    for i in range(len(seq) - len(separator) + 1):
        if torch.equal(seq[i:i + len(separator)], separator):  # Check if slice matches the separator
            last_index = i  # Update to the ENDING index of this match
    return last_index  # Return -1 if the separator is not found, or the last match index if found


def compute_expert_labels(seq, user_inst_seperator, data_seperator, response_seperator, num_labels: int = 3):
    expert_this = torch.zeros(len(seq), dtype=torch.long)  # Default: system instruction (0)

    if num_labels == 3:
        data_separator_position = find_first_occurrence(seq, data_seperator)
        if data_separator_position != -1:
            expert_this[data_separator_position:] = 1  # After the separator: data (1)

        response_separator_position = find_last_occurrence(seq, response_seperator)
        if response_separator_position != -1:
            expert_this[response_separator_position:] = 2  # After the separator: response (2)

    elif num_labels == 4:
        user_inst_seperator_position = find_first_occurrence(seq, user_inst_seperator)
        if user_inst_seperator_position != -1:
            expert_this[user_inst_seperator_position:] = 1  # After the separator: user instruction (1)

        data_separator_position = find_first_occurrence(seq, data_seperator)
        if data_separator_position != -1:
            expert_this[data_separator_position:] = 2  # After the separator: data (2)

        response_separator_position = find_last_occurrence(seq, response_seperator)
        if response_separator_position != -1:
            expert_this[response_separator_position:] = 3  # After the separator: response (2)

    else:
        raise ValueError("num_labels must be either 3 or 4")

    return expert_this


def _tokenize_fn(strings, tokenizer, frontend_delimiters, compute_gate=False, add_special_tokens=True):
    tokenizer.model_max_length = 16384
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=add_special_tokens
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]

    expert_labels = None
    if compute_gate:
        user_inst_seperator = tokenizer(
            DELIMITERS[frontend_delimiters][0],
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False
        ).input_ids[0]  # starting delimiter of data input

        data_seperator = tokenizer(
            DELIMITERS[frontend_delimiters][1],
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False
        ).input_ids[0]  # starting delimiter of data input

        response_seperator = tokenizer(
            DELIMITERS[frontend_delimiters][2],
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False
        ).input_ids[0]  # starting delimiter of response

        # Find the starting delimiters of Input
        expert_labels = []
        for seq in input_ids:
            expert_this = compute_expert_labels(seq, user_inst_seperator, data_seperator, response_seperator,
                                                num_labels=3)
            expert_labels.append(expert_this)

    input_ids_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    expert_labels_lens = labels_lens = input_ids_lens

    return dict(
        input_ids=input_ids,
        labels=labels,
        expert_labels=expert_labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
        expert_labels_lens=expert_labels_lens,
    )

def _tokenize_fn_batch(strings, tokenizer, frontend_delimiters, compute_gate=False, add_special_tokens=True):

    tokenizer.model_max_length = 1024

    if isinstance(strings, str):
        strings = [strings]

    # Standard HF batch tokenization: returns a BatchEncoding
    prompts_tok = tokenizer(
        strings,
        return_tensors="pt",
        padding=True,  # pad within the batch
        max_length=tokenizer.model_max_length,
        truncation=True,
        add_special_tokens=add_special_tokens,
    )

    input_ids      = prompts_tok.input_ids  # [B, L]
    attention_mask = prompts_tok.attention_mask  # [B, L]
    seq_lens = attention_mask.sum(dim=1).tolist()

    expert_labels = None

    if compute_gate:
        user_inst_seperator = tokenizer(
            DELIMITERS[frontend_delimiters][0],
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False,
        ).input_ids[0]

        data_seperator = tokenizer(
            DELIMITERS[frontend_delimiters][1],
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False,
        ).input_ids[0]

        response_seperator = tokenizer(
            DELIMITERS[frontend_delimiters][2],
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False,
        ).input_ids[0]

        # ---- NEW: clamp attention_mask up to response_seperator (inclusive) ----
        resp_ids = response_seperator.tolist()
        resp_len = len(resp_ids)

        for b, seq_len in enumerate(seq_lens):
            seq = input_ids[b, :seq_len].tolist()

            resp_end_idx = -1
            for i in range(seq_len - resp_len + 1):
                if seq[i:i + resp_len] == resp_ids:
                    resp_end_idx = i + resp_len - 1  # inclusive index
                    break

            if resp_end_idx != -1:
                # 1 up to response_seperator (inclusive), 0 afterwards
                attention_mask[b, :resp_end_idx + 1] = 1
                attention_mask[b, resp_end_idx + 1:] = 0

        max_len = input_ids.size(1)
        expert_list = []

        for seq, seq_len in zip(input_ids, seq_lens):
            seq_nopad = seq[:seq_len]
            # should return [seq_len]
            expert_this = compute_expert_labels(
                seq_nopad,
                user_inst_seperator,
                data_seperator,
                response_seperator,
                num_labels=3,
            )

            pad_len = max_len - expert_this.size(0)
            if pad_len > 0:
                pad = torch.full(
                    (pad_len,),
                    fill_value=2,  # ignore_index
                    dtype=expert_this.dtype,
                    device=expert_this.device,
                )
                expert_this = torch.cat([expert_this, pad], dim=0)

            expert_list.append(expert_this)

        expert_labels = torch.stack(expert_list, dim=0)  # [B, L]

    # IMPORTANT: Do NOT stuff lists into prompts_tok, they break .to()
    # Just return seq_lens alongside.

    return prompts_tok, expert_labels, seq_lens, attention_mask


def get_embedding_indices(tokenizer):
    init_values = [tokenizer.encode(v, add_special_tokens=False)[0]
                   for v in TEXTUAL_DELM_TOKENS] # get the delimiters tokens IDs
    ignore_values = [i for i in range(len(tokenizer))
                     if tokenizer.decode(i) == "#"] # get the padding token ID
    return init_values, ignore_values

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    REAL_DELIMITERS_INIT_EMBD_IND, _ = get_embedding_indices(tokenizer) # get embeddings for ['instruction', 'input',  'response', '###',  ':']

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens] = input_embeddings_avg # pad token's embedding is initialized to be average
        output_embeddings[-num_new_tokens] = output_embeddings_avg

        ## let ['[INST]', '[INPT]', '[RESP]', '[MARK]', '[COLN]'] = ['instruction', 'input',  'response', '###', ':']
        for i in range(len(SPECIAL_DELM_TOKENS)):
            input_embeddings[-num_new_tokens+i+1]  = input_embeddings[REAL_DELIMITERS_INIT_EMBD_IND[i]]
            output_embeddings[-num_new_tokens+i+1] = output_embeddings[REAL_DELIMITERS_INIT_EMBD_IND[i]]
