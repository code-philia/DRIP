#!/bin/bash

#CUDA_VISIBLE_DEVICES=5 python -m testing.test_overrefusal --model_name_or_path meta-llama/Llama-3.2-3B_SpclSpclSpcl-struq-none --data_path ./datasets/eval_data/over_refusal.json
#CUDA_VISIBLE_DEVICES=5 python -m testing.test_overrefusal --model_name_or_path meta-llama/Llama-3.2-3B_SpclSpclSpcl-struq --data_path ./datasets/eval_data/over_refusal.json
#CUDA_VISIBLE_DEVICES=5 python -m testing.test_overrefusal --model_name_or_path meta-llama/Llama-3.2-3B_SpclSpclSpcl-secalign --data_path ./datasets/eval_data/over_refusal.json
#CUDA_VISIBLE_DEVICES=5 python -m testing.test_overrefusal --model_name_or_path meta-llama/Llama-3.2-3B-SpclSpclSpcl-ise --pass_expert_labels --customized_model_class "LlamaForCausalLMMoE" --data_path ./datasets/eval_data/over_refusal.json
CUDA_VISIBLE_DEVICES=5 python -m testing.test_overrefusal --model_name_or_path meta-llama/Llama-3.2-3B-SpclSpclSpcl-instfuse --pass_expert_labels --customized_model_class "LlamaForCausalLMFuse" --data_path ./datasets/eval_data/over_refusal.json