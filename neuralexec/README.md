
# Step 1: Add model to the dataset

```commandline
cd neuralexec/
python add_llm_to_dataset.py --llm_name /../PromptInjection/meta-llama/Meta-Llama-3-8B-Instruct-TextTextText-instfuse-sep-none-newdata-dpo --delimiter TextTextText
```

# Step 2: Find the neuralexec triggers

```commandline
cd neuralexec/
python find_neuralexec.py confs.llama_instfuse --delimiter TextTextText
```

# Step 3:

```commandline
cd ../
./scripts/evaluation/llama8b/neuralexec.sh
```