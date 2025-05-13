
# Training on StruQ benchmark

## StruQ 
```bash
python -m torch.distributed.run --nproc_per_node=8 --master_port=29951 train.py \
--model_name_or_path meta-llama/Llama-3.2-3B --data_path datasets/alpaca_data_cleaned.json --output_dir meta-llama/Llama-3.2-3B_SpclSpclSpcl-struq \
--num_train_epochs 3 --bf16 True --per_device_train_batch_size 8 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 --evaluation_strategy "no" --save_strategy "epoch" \
--learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --attack SpclSpclSpcl_NaiveCompletion --model_max_length 512 --bf16 True --dataloader_num_workers 4 --fsdp "full_shard auto_wrap" --fsdp_config training/config/fsdp_config.json
```

## Secalign (fixme)
```bash
python -m torch.distributed.run --nproc_per_node=8 --master_port=29951 secalign.py \
--model_name_or_path meta-llama/Llama-3.2-3B_SpclSpclSpcl-struq --data_path datasets/alpaca_data_cleaned.json --output_dir meta-llama/Llama-3.2-3B_SpclSpclSpcl-secalign \
--num_train_epochs 1 --bf16 True --per_device_train_batch_size 8 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 --evaluation_strategy "no" --save_strategy "epoch" \
--learning_rate 2e-6 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --attack NaiveCompletion --model_max_length 512 --bf16 True --dataloader_num_workers 4 \
--alignment dpo
```

## ISE
```bash
python -m torch.distributed.run --nproc_per_node=8 --master_port=29951 train_ise.py \
--model_name_or_path meta-llama/Llama-3.2-3B --data_path datasets/alpaca_data_cleaned.json --output_dir meta-llama/Llama-3.2-3B-SpclSpclSpcl-ise \
--num_train_epochs 3 --per_device_train_batch_size 8 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 --evaluation_strategy "no" --save_strategy "epoch" \
--learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --attack SpclSpclSpcl_NaiveCompletion --model_max_length 512 --bf16 True --dataloader_num_workers 4 --fsdp "full_shard auto_wrap" --fsdp_config training/config/fsdp_config.json \
--extra_config_path training/config/Llama-3.2-3B-SpclSpclSpcl-ise.json 
```


## Ours
```bash
python -m torch.distributed.run --nproc_per_node=8 --master_port=29951 train_instfuse.py \
--model_name_or_path meta-llama/Llama-3.2-3B --data_path datasets/alpaca_data_cleaned.json --output_dir meta-llama/Llama-3.2-3B-SpclSpclSpcl-instfuse \
--num_train_epochs 3 --per_device_train_batch_size 8 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 --evaluation_strategy "no" --save_strategy "epoch" \
--learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --attack SpclSpclSpcl_NaiveCompletion --model_max_length 512 --bf16 True --dataloader_num_workers 4 --fsdp "full_shard auto_wrap" --fsdp_config training/config/fsdp_config.json
```

# Training on Instruction Hierarchy benchmark

## StruQ 
```bash
python -m torch.distributed.run --nproc_per_node=8 --master_port=29951 train_hier.py \
--model_name_or_path meta-llama/Llama-3.2-3B --data_path datasets/InstructHierarchy/ultrachat-190K-final.json datasets/InstructHierarchy/data_instruction_10k.json datasets/InstructHierarchy/long_prompt_extract-10k.json datasets/InstructHierarchy/long_prompt_follow-10k.json datasets/InstructHierarchy/long_prompt_ori-10k.json datasets/InstructHierarchy/ultrachat-10k-split-final.json  datasets/InstructHierarchy/user_change_system-10k.json datasets/InstructHierarchy/user_conflict_system-10k.json datasets/InstructHierarchy/user_follow_system-10k.json \
--output_dir meta-llama-hier/Llama-3.2-3B_SpclSpclSpcl-struq \
--num_train_epochs 3 --per_device_train_batch_size 8 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 --evaluation_strategy "no" --save_strategy "epoch" \
--learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --attack SpclSpclSpcl_None --model_max_length 512 --bf16 True --dataloader_num_workers 4 --fsdp "full_shard auto_wrap" --fsdp_config training/config/fsdp_config.json
```

## ISE
```bash
python -m torch.distributed.run --nproc_per_node=8 --master_port=29951 train_ise_hier.py \
--model_name_or_path meta-llama/Llama-3.2-3B --data_path datasets/InstructHierarchy/ultrachat-190K-final.json datasets/InstructHierarchy/data_instruction_10k.json datasets/InstructHierarchy/long_prompt_extract-10k.json datasets/InstructHierarchy/long_prompt_follow-10k.json datasets/InstructHierarchy/long_prompt_ori-10k.json datasets/InstructHierarchy/ultrachat-10k-split-final.json  datasets/InstructHierarchy/user_change_system-10k.json datasets/InstructHierarchy/user_conflict_system-10k.json datasets/InstructHierarchy/user_follow_system-10k.json \
--output_dir meta-llama-hier/Llama-3.2-3B-SpclSpclSpcl-ise \
--num_train_epochs 3 --per_device_train_batch_size 8 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 --evaluation_strategy "no" --save_strategy "epoch" \
--learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --attack SpclSpclSpcl_None --model_max_length 512 --bf16 True --dataloader_num_workers 4 --fsdp "full_shard auto_wrap" --fsdp_config training/config/fsdp_config.json \
--extra_config_path training/config/Llama-3.2-3B-SpclSpclSpcl-ise.json 
```

## Ours
```bash
python -m torch.distributed.run --nproc_per_node=8 --master_port=29951 train_instfuse_hier.py \
--model_name_or_path meta-llama/Llama-3.2-3B --data_path datasets/InstructHierarchy/ultrachat-190K-final.json datasets/InstructHierarchy/data_instruction_10k.json datasets/InstructHierarchy/long_prompt_extract-10k.json datasets/InstructHierarchy/long_prompt_follow-10k.json datasets/InstructHierarchy/long_prompt_ori-10k.json datasets/InstructHierarchy/ultrachat-10k-split-final.json  datasets/InstructHierarchy/user_change_system-10k.json datasets/InstructHierarchy/user_conflict_system-10k.json datasets/InstructHierarchy/user_follow_system-10k.json \
--output_dir meta-llama-hier/Llama-3.2-3B-SpclSpclSpcl-instfuse \
--num_train_epochs 3 --per_device_train_batch_size 8 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 --evaluation_strategy "no" --save_strategy "epoch" \
--learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --attack SpclSpclSpcl_None --model_max_length 512 --bf16 True --dataloader_num_workers 4 --fsdp "full_shard auto_wrap" --fsdp_config training/config/fsdp_config.json
```


# Evaluation

## No defence
```bash
python -m testing.test --model_name_or_path meta-llama/Llama-3.2-3B \
--attack none naive ignore completion_real escape_separation
```

## StruQ
```bash
python -m testing.test --model_name_or_path meta-llama/Llama-3.2-3B_SpclSpclSpcl-struq \
--attack none naive ignore completion_real escape_separation
```

## ISE
```bash
python -m testing.test --model_name_or_path meta-llama/Llama-3.2-3B-SpclSpclSpcl-ise  \
--attack none naive ignore completion_real escape_separation \
--pass_expert_labels --customized_model_class "LlamaForCausalLMMoE"
```

## Ours
```bash
python -m testing.test --model_name_or_path meta-llama/Llama-3.2-3B-SpclSpclSpcl-instfuse  \
--attack none naive ignore completion_real escape_separation \
--pass_expert_labels --customized_model_class "LlamaForCausalLMFuse"
```

# Evaluation GCG
## StruQ
```bash
python -m testing.test_gcg --model_name_or_path meta-llama/Llama-3.2-3B_SpclSpclSpcl-struq
```

## Secalign
```bash
python -m testing.test_gcg --model_name_or_path meta-llama/Llama-3.2-3B_SpclSpclSpcl-secalign
```

## ISE
```bash
python -m testing.test_gcg --model_name_or_path meta-llama/Llama-3.2-3B-SpclSpclSpcl-ise \
--pass_expert_labels --customized_model_class "LlamaForCausalLMMoE"
```

## Ours
```bash
python -m testing.test_gcg --model_name_or_path meta-llama/Llama-3.2-3B-SpclSpclSpcl-instfuse \
--pass_expert_labels --customized_model_class "LlamaForCausalLMFuse"
```

# Evaluation NeuralExec
