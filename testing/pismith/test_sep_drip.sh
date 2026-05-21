#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate prompt  # Replace with your actual env name

CUDA_VISIBLE_DEVICES=1 python -m testing.pismith.test_pismith_sep \
    -m meta-llama/Llama-3.1-8B-Instruct-log-TextTextText-instfuse-alpaca-dpo \
    --customized_model_class LlamaForCausalLMDRIP \
    --attack_model_path ./pismith_ckpt/alpaca/attack_lm_final