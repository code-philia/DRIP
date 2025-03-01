
# Generate model answer

```commandline
 python -m testing.mt_bench.gen_model_answer --model-path meta-llama/Llama-3.2-1B-SpclSpclSpcl_NaiveCompletion-struq --model-id Llama-3.2-1B-SpclSpclSpcl_NaiveCompletion-struq 
```

```commandline
python -m testing.mt_bench.gen_model_answer --model-path meta-llama/Llama-3.2-1B-SpclSpclSpcl_NaiveCompletion-instfuse --model-id Llama-3.2-1B-SpclSpclSpcl_NaiveCompletion-instfuse --pass_expert_labels --customized_model_class LlamaForCausalLMFuse
```

```commandline
python -m testing.mt_bench.gen_model_answer --model-path meta-llama/Llama-3.2-1B-SpclSpclSpcl_NaiveCompletion-ise --model-id Llama-3.2-1B-SpclSpclSpcl_NaiveCompletion-ise --pass_expert_labels --customized_model_class LlamaForCausalLMMoE
```

# Scoring model answer
```commandline

```

# Visualize score
