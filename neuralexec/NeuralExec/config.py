# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

IGNORE_INDEX        = -100
DEFAULT_TOKENS      = {'pad_token': '[PAD]', 'eos_token': '</s>', 'bos_token': '<s>', 'unk_token': '<unk>'}
TEXTUAL_DELM_TOKENS = ['instruction', 'input',  'response', '###',    ':']
SPECIAL_DELM_TOKENS = ['[INST]', '[INPT]', '[RESP]', '[MARK]', '[COLN]']
FILTERED_TOKENS     = SPECIAL_DELM_TOKENS + ['##']
OTHER_DELM_TOKENS   = {
                        'mark': ['{s}', '|{s}|', '<{s}>', '[{s}]', '<|{s}|>', '[|{s}|]', '<[{s}]>', '\'\'\'{s}\'\'\'', '***{s}***'],
                        'inst': ['Command', 'Rule', 'Prompt', 'Task'],
                        'inpt': ['Data', 'Context', 'Text'],
                        'resp': ['Output', 'Answer', 'Reply'],
                        'user': ['', 'Prompter ', 'User ', 'Human '],
                        'asst': ['', 'Assistant ', 'Chatbot ', 'Bot ', 'GPT ', 'AI '],
                       }
OTHER_DELM_FOR_TEST = 2

DELIMITERS = {
    "TextTextText": ['<|begin_of_text|><|start_header_id|>system<|end_header_id|>',
                     '<|eot_id|><|start_header_id|>user<|end_header_id|>',
                     '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'],
    "TextTextTextMistral": ['<s>[INST] <<SYS>>', ' <</SYS>>', '[/INST]'],
    "SpclSpclSpcl": [SPECIAL_DELM_TOKENS[3] + ' ' + SPECIAL_DELM_TOKENS[0] + SPECIAL_DELM_TOKENS[4],
                     SPECIAL_DELM_TOKENS[3] + ' ' + SPECIAL_DELM_TOKENS[1] + SPECIAL_DELM_TOKENS[4],
                     SPECIAL_DELM_TOKENS[3] + ' ' + SPECIAL_DELM_TOKENS[2] + SPECIAL_DELM_TOKENS[4]],
    "Mistral-7B-Instruct-v0.3-log": ['<s>[INST] <<SYS>>', ' <</SYS>>', '[/INST]'],
    "Meta-Llama-3-8B-Instruct-log":
        ['<|begin_of_text|><|start_header_id|>system<|end_header_id|>',
         '<|eot_id|><|start_header_id|>user<|end_header_id|>',
         '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'],
    "Qwen2.5-7B-Instruct-log":
        ['<|im_start|>system',
         '<|im_end|><|im_start|>user',
         '<|im_end|><|im_start|>assistant'],
}

PROMPT_FORMAT = {}
for name, delm in DELIMITERS.items():
    sys_input = ''
    sys_no_input = ''
    PROMPT_FORMAT[name] = {}
    PROMPT_FORMAT[name]["prompt_input"]    = sys_input    + delm[0] + "\n{instruction}\n\n" + delm[1] + "\n{input}\n\n" + delm[2] + "\n"
    PROMPT_FORMAT[name]["prompt_no_input"] = sys_no_input + delm[0] + "\n{instruction}\n\n" + delm[2] + "\n"

