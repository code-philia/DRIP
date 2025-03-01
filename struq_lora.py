# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import transformers
from transformers import Trainer
from config import IGNORE_INDEX, DEFAULT_TOKENS, SPECIAL_DELM_TOKENS, TEXTUAL_DELM_TOKENS
from train import ModelArguments, DataArguments,TrainingArguments, AttackArguments, make_supervised_data_module_orig, smart_tokenizer_and_embedding_resize
from peft import LoraConfig, get_peft_model

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, AttackArguments))
    model_args, data_args, training_args, attack_args = parser.parse_args_into_dataclasses()
    data_args.attack = attack_args.attack 
    if 'Instruct' in model_args.model_name_or_path:
        assert 'SpclSpclSpcl' not in data_args.attack
    print('\n\n' + training_args.output_dir + '\n\n')

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    if model_args.window_size > 0:
        model.config.window = model_args.window_size

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
    )

    special_tokens_dict = dict()
    special_tokens_dict["pad_token"] = DEFAULT_TOKENS['pad_token'] ###
    special_tokens_dict["eos_token"] = DEFAULT_TOKENS['eos_token']
    special_tokens_dict["bos_token"] = DEFAULT_TOKENS['bos_token']
    special_tokens_dict["unk_token"] = DEFAULT_TOKENS['unk_token']
    special_tokens_dict["additional_special_tokens"] = SPECIAL_DELM_TOKENS ### 
    smart_tokenizer_and_embedding_resize(special_tokens_dict=special_tokens_dict, tokenizer=tokenizer, model=model)

    data_module = make_supervised_data_module_orig(tokenizer=tokenizer, data_args=data_args, downsample=training_args.downsample)
    if not training_args.downsample and training_args.lr_scale:
        training_args.learning_rate /= data_module["train_dataset"].data_copy_count

    ## fixme: for a fair comparison, I still use lora
    lora_config = LoraConfig(
        r=16,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=["q_proj", "v_proj", "o_proj"],
        modules_to_save=["lm_head"],  # fixme
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if "lm_head" in name:
            param.requires_grad = True  # ensure that they are not frozen

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"✅ {name} is trainable")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

    
if __name__ == "__main__":
    train()
    # model = "huggyllama/llama-7b"
    # data_path = "datasets/davinci_003_outputs.json"
    # attack = "SpclSpclSpcl_NaiveCompletion"
    # model_max_length = 512
    # lr = 2e-5

    # --model_name_or_path meta-llama/Llama-3.2-1B --data_path datasets/alpaca_data_cleaned.json --bf16 True  --output_dir debug  --num_train_epochs 3  --per_device_train_batch_size 2 --per_device_eval_batch_size 2
    #             --gradient_accumulation_steps 8  --evaluation_strategy "no"  --save_strategy "no"  --save_total_limit 1  --learning_rate 2e-5  --weight_decay 0.  --warmup_ratio 0.03 \
    #             --lr_scheduler_type "cosine" --logging_steps 1 \
    #             --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" \
    #             --tf32 True --attack SpclSpclSpcl_NaiveCompletion --model_max_length 512