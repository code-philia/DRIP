
# Generate model answer

Struq
```commandline
python -m testing.mt_bench.gen_model_answer \
--model-path meta-llama/Llama-3.2-3B_SpclSpclSpcl-struq --model-id Llama-3.2-1B-SpclSpclSpcl_NaiveCompletion-struq
```

ISE
```commandline
python -m testing.mt_bench.gen_model_answer \
--model-path meta-llama/Llama-3.2-3B-SpclSpclSpcl-ise --model-id Llama-3.2-1B-SpclSpclSpcl_NaiveCompletion-ise \
--pass_expert_labels --customized_model_class "LlamaForCausalLMMoE"
```



# Scoring model answer
```bash
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e ".[model_worker,llm_judge]"
```

```bash
export OPENAI_API_KEY=XXXX
```

```bash
python -m testing.mt_bench.gen_judgment \
--model-path meta-llama/Llama-3.2-3B-SpclSpclSpcl-instfuse --model-id Llama-3.2-3B-SpclSpclSpcl_NaiveCompletion-instfuse --parallel 8
```

# Visualize score
