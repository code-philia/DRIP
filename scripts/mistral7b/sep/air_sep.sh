#!/bin/bash

SCRIPT_PATH="train_unified.py"
BASELINE="air"
BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
DATA_PATH="datasets/sep/sep_data_dpo.json"
FILENAME=$(basename "$DATA_PATH")
PREFIX=${FILENAME%%_*}
FSDP_CONFIG="training/config/fsdp_config_mistral.json"
DELIMITER="TextTextTextMistral"

SAVE_PATH="${BASE_MODEL}-${DELIMITER}-${BASELINE}-${PREFIX}-none"

BATCH_SIZE=1
EPOCH=3

OBJECTIVE="air_dpo"
MODEL_FAMILY="mistral"
ARCH="air"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 \
python -m torch.distributed.run --nproc_per_node=8 --master_port=29951 "$SCRIPT_PATH" \
  --objective "${OBJECTIVE}" \
  --model-family "${MODEL_FAMILY}" \
  --arch "${ARCH}" \
  --model_name_or_path "$BASE_MODEL" \
  --data_path "$DATA_PATH" \
  --output_dir "$SAVE_PATH" \
  --num_train_epochs "$EPOCH" \
  --bf16 True \
  --gradient_checkpointing True \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --save_strategy "steps" \
  --save_steps 10 \
  --save_total_limit 3 \
  --learning_rate 1e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --attack "${DELIMITER}_None" \
  --model_max_length 256 \
  --dataloader_num_workers 4 \
  --optim "paged_adamw_32bit" \
  --resume_from_checkpoint True