#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate prompt  # Replace with your actual env name

CUDA_VISIBLE_DEVICES=0 python -m testing.pismith.test_pismith_sep \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct-log \
    --attack_model_path ./pismith_ckpt/alpaca_llama/attack_lm_final