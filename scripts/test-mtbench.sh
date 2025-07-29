#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python -m testing.mt_bench.gen_model_answer --model_name_or_path meta-llama/Llama-3.2-3B_SpclSpclSpcl-struq-none --model-id Llama-3B-Baseline
CUDA_VISIBLE_DEVICES=7 python -m testing.mt_bench.gen_model_answer --model_name_or_path meta-llama/Llama-3.2-3B_SpclSpclSpcl-struq --model-id Llama-3B-Struq
CUDA_VISIBLE_DEVICES=7 python -m testing.mt_bench.gen_model_answer --model_name_or_path meta-llama/Llama-3.2-3B_SpclSpclSpcl-secalign --model-id Llama-3B-Secalign
CUDA_VISIBLE_DEVICES=7 python -m testing.mt_bench.gen_model_answer --model_name_or_path meta-llama/Llama-3.2-3B-SpclSpclSpcl-ise --pass_expert_labels --customized_model_class "LlamaForCausalLMMoE" --model-id Llama-3B-ISE
CUDA_VISIBLE_DEVICES=7 python -m testing.mt_bench.gen_model_answer --model_name_or_path meta-llama/Llama-3.2-3B-SpclSpclSpcl-instfuse --pass_expert_labels --customized_model_class "LlamaForCausalLMFuse" --model-id Llama-3B-Ours