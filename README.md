# DRIP

Official code for **"DRIP: Defending Prompt Injection via Token-wise Representation Editing and Residual Fusion."**

## Overview

DRIP introduces two architectural modifications:

- A **token-wise de-instruction shift** that moves the representation of data tokens away from directive semantics.
- A **residual re-instruction fusion** path that persistently anchors generation on the top-level instruction.

![Overview](figures/overview.png)

---

# Setup

## 1. Download base model checkpoints

```bash
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 \
    --local-dir mistralai/Mistral-7B-Instruct-v0.3 \
    --resume-download --local-dir-use-symlinks False

huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \
    --local-dir meta-llama/Meta-Llama-3-8B-Instruct \
    --resume-download --local-dir-use-symlinks False
```

## 2. Create the environment

Run the setup script. 
It creates a conda environment (default name `prompt`) and installs all pinned dependencies.

```bash
bash setup_env.sh
conda activate prompt
```

## 3. Select GPUs

Training uses FSDP across 6 NVIDIA RTX 5880 GPUs. 
Export the visible devices accordingly:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
```


# Data Curation
 
Data curation is already done. The curated dataset is uploaded at [`datasets/sep/sep_data_cleaned_dpo_gpt.json`](./datasets/sep/sep_data_cleaned_dpo_gpt.json).
 
To generate the DRIP training data from scratch, see [`data_generation/README.md`](./data_generation/README.md).
 
---

# Training

With base model as **Meta-Llama-3-8B-Instruct**:

```bash
bash ./scripts/llama8b/sep/drip_sep.sh
```

Alternatively with base model as **Mistral-7B-Instruct-v0.3**:

```bash
bash ./scripts/mistral7b/sep/drip_sep.sh
```

---

# Evaluation

Before running any evaluation, copy [`./datasets/openai_configs_example.yaml`](./datasets/openai_configs_example.yaml) to `./datasets/openai_configs.yaml` and fill in your OpenAI configuration.

All evaluation scripts prompt for a single CUDA device ID and the trained model path. The examples below use the Llama scripts; swap `llama8b` for `mistral7b` for the other model.

## SEP score

1. Run [`./scripts/evaluation/llama8b/sep.sh`](./scripts/evaluation/llama8b/sep.sh).
2. When prompted, enter the CUDA device ID and the model path.
3. Run [`./testing/sep/evaluation_main.py`](./testing/sep/evaluation_main.py) to print the SEP scores.

## ASR

**Alpaca heuristic-based attacks**

1. Run [`./scripts/evaluation/llama8b/alpaca_injection.sh`](./scripts/evaluation/llama8b/alpaca_injection.sh).
2. When prompted, enter the CUDA device ID and the model path.
3. Run [`./testing/evaluation_main.py`](./testing/evaluation_main.py) to print ASR under the Naive, Ignore, Completion, Escape, and HackaPrompt attacks.

**GCG-based adaptive attacks**

See [`gcg/README.md`](./gcg/README.md). GCG requires a separate legacy environment because newer `transformers` versions trigger OOM.

**InjecAgent**

1. Run [`./scripts/evaluation/llama8b/injecagent.sh`](./scripts/evaluation/llama8b/injecagent.sh).
2. When prompted, enter the CUDA device ID and the model path.

## Utility

**AlpacaEval 2.0** (can cost up to USD 50)

1. Run [`./scripts/evaluation/llama8b/alpaca_utility.sh`](./scripts/evaluation/llama8b/alpaca_utility.sh).
2. When prompted, enter the CUDA device ID and the model path.
3. If the `alpacaeval` command is not runned successfully, please manually do 

```bash
export OPENAI_CLIENT_CONFIG_PATH=./datasets/openai_configs.yaml && alpaca_eval --model_outputs [model_path]/predictions_on_davinci_003_outputs.json --reference_outputs datasets/gpt4o_outputs.json
```

4. Find the win rate in `model-path/weighted_alpaca_eval_gpt4_turbo/leaderboard.csv`.

**IFEval**

1. Run [`./scripts/evaluation/llama8b/ifeval.sh`](./scripts/evaluation/llama8b/ifeval.sh).
2. When prompted, enter the CUDA device ID and the model path.
3. Run [`./testing/ifeval/evaluation_main.py`](./testing/ifeval/evaluation_main.py) and look for ASR strict.

**MT-Bench**

1. Run [`./scripts/evaluation/llama8b/mtbench.sh`](./scripts/evaluation/llama8b/mtbench.sh).
2. When prompted, enter the CUDA device ID and the model path.
3. Run [`./testing/mt_bench/gen_judgment.py`](./testing/mt_bench/gen_judgment.py) with `--model-path [model-path] --model-id [model name, e.g. Ours]`.
4. Plot the radar chart with [`./testing/mt_bench/plot.py`](./testing/mt_bench/plot.py).

**MMLU**

1. Run [`./scripts/evaluation/llama8b/mmlu_utility.sh`](./scripts/evaluation/llama8b/mmlu_utility.sh).
2. When prompted, enter the CUDA device ID and the model path.
3. Run [`./testing/mmlu/evaluation_main.py`](./testing/mmlu/evaluation_main.py).