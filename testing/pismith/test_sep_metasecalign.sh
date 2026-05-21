#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate prompt  # Replace with your actual env name

CUDA_VISIBLE_DEVICES=1 python -m testing.pismith.test_pismith_sep \
     -m meta-llama/Meta-SecAlign-8B-merged \
    --attack_model_path ./pismith_ckpt/alpaca_metasecalign/attack_lm_final