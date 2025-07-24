
# CLEAN Training on StruQ benchmark

# StruQ 

## meta-llama/Llama-3.2-3B
```bash
python -m torch.distributed.run --nproc_per_node=8 --master_port=29951 train.py \
--model_name_or_path meta-llama/Llama-3.2-3B --data_path datasets/alpaca_data_cleaned.json datasets/sep/sep_data_cleaned.json \
--output_dir meta-llama/Llama-3.2-3b-SpclSpclSpcl-struq-sep-none \
--num_train_epochs 3 --bf16 True --per_device_train_batch_size 4 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8  --save_strategy "epoch" \
--learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --attack SpclSpclSpcl_None --model_max_length 512 --bf16 True --dataloader_num_workers 4 \
--fsdp "full_shard auto_wrap" --fsdp_config training/config/fsdp_config.json
```

## ministral/Ministral-3b-instruct
```bash
python -m torch.distributed.run --nproc_per_node=8 --master_port=29951 train.py \
--model_name_or_path ministral/Ministral-3b-instruct --data_path datasets/alpaca_data_cleaned.json datasets/sep/sep_data_cleaned.json \
--output_dir ministral/ministral-3b-SpclSpclSpcl-struq-sep-none \
--num_train_epochs 3 --bf16 True --per_device_train_batch_size 4 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8  --save_strategy "epoch" \
--learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --attack SpclSpclSpcl_None --model_max_length 512 --bf16 True --dataloader_num_workers 4 \
--fsdp "full_shard auto_wrap" --fsdp_config training/config/fsdp_config_mistral.json
```

## llama8b


# ISE
## meta-llama/Llama-3.2-3B
```bash
python -m torch.distributed.run --nproc_per_node=8 --master_port=29951 train_ise.py \
--model_name_or_path meta-llama/Llama-3.2-3B --data_path datasets/alpaca_data_cleaned.json datasets/sep/sep_data_cleaned.json \
--output_dir meta-llama/Llama-3.2-3b-SpclSpclSpcl-ise-sep-none \
--num_train_epochs 3 --per_device_train_batch_size 4 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8  --save_strategy "epoch" \
--learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --attack SpclSpclSpcl_None --model_max_length 512 --bf16 True --dataloader_num_workers 4 \
--fsdp "full_shard auto_wrap" --fsdp_config training/config/fsdp_config.json
```

## ministral/Ministral-3b-instruct
```bash
python -m torch.distributed.run --nproc_per_node=8 --master_port=29951 train_ise_mistral.py \
--model_name_or_path ministral/Ministral-3b-instruct --data_path datasets/alpaca_data_cleaned.json datasets/sep/sep_data_cleaned.json \
--output_dir ministral/ministral-3b-SpclSpclSpcl-ise-sep-none \
--num_train_epochs 3 --per_device_train_batch_size 4 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8  --save_strategy "epoch" \
--learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --attack SpclSpclSpcl_None --model_max_length 512 --bf16 True --dataloader_num_workers 4 \
--fsdp "full_shard auto_wrap" --fsdp_config training/config/fsdp_config_mistral.json
```

# POSSEP
## meta-llama/Llama-3.2-3B
```bash
python -m torch.distributed.run --nproc_per_node=8 --master_port=29951 train_possep.py \
--model_name_or_path meta-llama/Llama-3.2-3B --data_path datasets/alpaca_data_cleaned.json datasets/sep/sep_data_cleaned.json \
--output_dir meta-llama/Llama-3.2-3b-SpclSpclSpcl-possep-sep-none \
--num_train_epochs 3 --per_device_train_batch_size 4 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8  --save_strategy "epoch" \
--learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --attack SpclSpclSpcl_None --model_max_length 512 --bf16 True --dataloader_num_workers 4 \
--fsdp "full_shard auto_wrap" --fsdp_config training/config/fsdp_config.json
```

## ministral/Ministral-3b-instruct
```bash
python -m torch.distributed.run --nproc_per_node=8 --master_port=29951 train_possep_mistral.py \
--model_name_or_path ministral/Ministral-3b-instruct --data_path datasets/alpaca_data_cleaned.json datasets/sep/sep_data_cleaned.json \
--output_dir ministral/ministral-3b-SpclSpclSpcl-possep-sep-none \
--num_train_epochs 3 --per_device_train_batch_size 4 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8  --save_strategy "epoch" \
--learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --attack SpclSpclSpcl_None --model_max_length 512 --bf16 True --dataloader_num_workers 4 \
--fsdp "full_shard auto_wrap" --fsdp_config training/config/fsdp_config_mistral.json
```

# InstFuse
## meta-llama/Llama-3.2-3B
```bash
python -m torch.distributed.run --nproc_per_node=8 --master_port=29951 train_instfuse.py \
--model_name_or_path meta-llama/Llama-3.2-3B --data_path datasets/alpaca_data_cleaned.json datasets/sep/sep_data_cleaned.json \
--output_dir meta-llama/Llama-3.2-3b-SpclSpclSpcl-instfuse-sep-none \
--num_train_epochs 3 --per_device_train_batch_size 6 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8  --save_strategy "epoch" \
--learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --attack SpclSpclSpcl_None --model_max_length 512 --bf16 True --dataloader_num_workers 4 \
--fsdp "full_shard auto_wrap" --fsdp_config training/config/fsdp_config.json
```

## ministral/Ministral-3b-instruct
```bash
python -m torch.distributed.run --nproc_per_node=8 --master_port=29951 train_instfuse_mistral.py \
--model_name_or_path ministral/Ministral-3b-instruct --data_path datasets/alpaca_data_cleaned.json datasets/sep/sep_data_cleaned.json \
--output_dir ministral/ministral-3b-SpclSpclSpcl-instfuse-sep-none \
--num_train_epochs 3 --per_device_train_batch_size 4 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8  --save_strategy "epoch" \
--learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --attack SpclSpclSpcl_None --model_max_length 512 --bf16 True --dataloader_num_workers 4 \
--fsdp "full_shard auto_wrap" --fsdp_config training/config/fsdp_config_mistral.json
```

# CLEAN Training on Instruction Hierarchy benchmark

## StruQ 
```bash
python -m torch.distributed.run --nproc_per_node=8 --master_port=29951 train_hier.py \
--model_name_or_path meta-llama/Llama-3.2-3B \
--data_path datasets/InstructHierarchy/long_prompt_follow-10k.json datasets/InstructHierarchy/long_prompt_ori-10k.json datasets/InstructHierarchy/ultrachat-10k-split-final.json  datasets/InstructHierarchy/user_follow_system-10k.json \
--output_dir meta-llama-hier/Llama-3.2-3b-SpclSpclSpcl-struq-clean \
--num_train_epochs 3 --per_device_train_batch_size 8 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8  --save_strategy "epoch" \
--learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --attack SpclSpclSpcl_None --model_max_length 512 --bf16 True --dataloader_num_workers 4 --fsdp "full_shard auto_wrap" --fsdp_config training/config/fsdp_config.json
```

## ISE
```bash
python -m torch.distributed.run --nproc_per_node=8 --master_port=29951 train_ise_hier.py \
--model_name_or_path meta-llama/Llama-3.2-3B \
--data_path datasets/InstructHierarchy/long_prompt_follow-10k.json datasets/InstructHierarchy/long_prompt_ori-10k.json datasets/InstructHierarchy/ultrachat-10k-split-final.json  datasets/InstructHierarchy/user_follow_system-10k.json \
--output_dir meta-llama-hier/Llama-3.2-3b-SpclSpclSpcl-ise-clean \
--num_train_epochs 3 --per_device_train_batch_size 8 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8  --save_strategy "epoch" \
--learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --attack SpclSpclSpcl_None --model_max_length 512 --bf16 True --dataloader_num_workers 4 --fsdp "full_shard auto_wrap" --fsdp_config training/config/fsdp_config.json
```

## Possep
```bash
python -m torch.distributed.run --nproc_per_node=8 --master_port=29951 train_possep_hier.py \
--model_name_or_path meta-llama/Llama-3.2-3B \
--data_path datasets/InstructHierarchy/long_prompt_follow-10k.json datasets/InstructHierarchy/long_prompt_ori-10k.json datasets/InstructHierarchy/ultrachat-10k-split-final.json  datasets/InstructHierarchy/user_follow_system-10k.json \
--output_dir meta-llama-hier/Llama-3.2-3b-SpclSpclSpcl-possep-clean \
--num_train_epochs 3 --per_device_train_batch_size 8 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8  --save_strategy "epoch" \
--learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --attack SpclSpclSpcl_None --model_max_length 512 --bf16 True --dataloader_num_workers 4 --fsdp "full_shard auto_wrap" --fsdp_config training/config/fsdp_config.json
```

## Ours
```bash
python -m torch.distributed.run --nproc_per_node=8 --master_port=29951 train_instfuse_hier.py \
--model_name_or_path meta-llama/Llama-3.2-3B \
--data_path datasets/InstructHierarchy/long_prompt_follow-10k.json datasets/InstructHierarchy/long_prompt_ori-10k.json datasets/InstructHierarchy/ultrachat-10k-split-final.json  datasets/InstructHierarchy/user_follow_system-10k.json \
--output_dir meta-llama-hier/Llama-3.2-3b-SpclSpclSpcl-instfuse-clean \
--num_train_epochs 3 --per_device_train_batch_size 8 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8  --save_strategy "epoch" \
--learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --attack SpclSpclSpcl_None --model_max_length 512 --bf16 True --dataloader_num_workers 4 --fsdp "full_shard auto_wrap" --fsdp_config training/config/fsdp_config.json
```


# Evaluation

## StruQ
```bash
python -m testing.test --model_name_or_path meta-llama/Llama-3.2-3b-SpclSpclSpcl-struq-none \
--attack none naive ignore_0 completion_real escape_separation
```

## ISE
```bash
python -m testing.test --model_name_or_path meta-llama/Llama-3.2-3b-SpclSpclSpcl-ise-none  \
--attack none naive ignore_0 completion_real escape_separation \
--pass_expert_labels --customized_model_class "LlamaForCausalLMMoE"
```

## PosSep
```bash
python -m testing.test --model_name_or_path meta-llama/Llama-3.2-3b-SpclSpclSpcl-possep-none \
--attack none naive ignore_0 completion_real escape_separation \
--pass_expert_labels --customized_model_class "LlamaForCausalLMMoEV2"
```

## Ours
```bash
python -m testing.test --model_name_or_path meta-llama/Llama-3.2-3b-SpclSpclSpcl-instfuse-none  \
--attack none naive ignore_0 completion_real escape_separation \
--pass_expert_labels --customized_model_class "LlamaForCausalLMFuse"
```

# GCG Attack
## StruQ
```bash
python -m testing.test_gcg --model_name_or_path meta-llama/Llama-3.2-3B-SpclSpclSpcl-struq
```

## Secalign
```bash
python -m testing.test_gcg --model_name_or_path meta-llama/Llama-3.2-3B-SpclSpclSpcl-secalign
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

# Test on SEP
```bash
python -m testing.sep.test_sep --model_name_or_path meta-llama/Llama-3.2-3b-SpclSpclSpcl-instfuse-none \
--pass_expert_labels --customized_model_class "LlamaForCausalLMFuse"
```

