# Introduction

## Motivation: 

LLMs are vulnerable to prompt injection, and existing defenses often remove all instruction-like data. While robust, this approach loses information when such content is indeed part of the data. 
We thus argue for **de-instructing**: suppressing directive intent without discarding information.
![Motivating example.](figures/motivating.png)

## Contributions

DRIP introduces two architectural modifications: 
- A **token-wise de-instruction shift**, which adjusts the representation of data tokens away from directive semantics, and 
- A **residual re-instruction fusion** path, which persistently anchors the model's generation on the top-level instruction. 

![Overview.](figures/overview.png)
---

# Setup
## Download base model checkpoints
```bash
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --local-dir mistralai/Mistral-7B-Instruct-v0.3-log --resume-download  --local-dir-use-symlinks False
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir meta-llama/Meta-Llama-3-8B-Instruct-log --resume-download  --local-dir-use-symlinks False
huggingface-cli download TinyLlama/TinyLlama-1.1B-step-50K-105b --local-dir TinyLlama/TinyLlama-1.1B-step-50K-105b --resume-download  --local-dir-use-symlinks False
```

## Training data curation

| Data file                                                                                                                                                   |                                        Explanation                                        |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------:|
| [./datasets/sep/sep_data_cleaned_dpo_gpt.json](./datasets/sep/sep_data_cleaned_dpo_gpt.json)                                                                |                     Our newly curated training data for DPO learning                      |
| [./datasets/sep/sep_data_cleaned_sft_gpt.json](./datasets/sep/sep_data_cleaned_sft_gpt.json)                                                                | Our newly curated training data with standard SFT learning (without contrastive learning) |
| [./datasets/sep/sep_data_origdata_dpo.json](datasets/sep/sep_data_origdata_dpo.json) and [./datasets/sep/sep_data_dpo.json](datasets/sep/sep_data_dpo.json) |                        SEP original training data for DPO learning                        | 
| [./datasets/sep/sep_data_cleaned.json](datasets/sep/sep_data_cleaned.json)                                                                                                                    |                        SEP original training data for standard SFT learning                        |       

## Install pre-requisites

- Create conda environment and install the appropriate transformers version
```bash
conda create -n prompt python=3.10
conda activate prompt
pip install -r requirements.txt
```
- I am using 6 NVIDIA RTX 5880 GPU devices for FSDP training
Export CUDA_VISIBLE_DEVICES environment variable ```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6```

---

# Training

## For Meta-Llama-3-8B-Instruct
- Training with StruQ [./scripts/llama8b/sep/struq_sep.sh](./scripts/llama8b/sep/struq_sep.sh)
- Training with SecAlign [./scripts/llama8b/sep/secalign_sep.sh](./scripts/llama8b/sep/secalign_sep.sh)
- Training with ISE [./scripts/llama8b/sep/ise_sep.sh](./scripts/llama8b/sep/ise_sep.sh)
- Training with PFT [./scripts/llama8b/sep/pft_sep.sh](./scripts/llama8b/sep/pft_sep.sh)
- Training with DRIP (Ours) [./scripts/llama8b/sep/instfuse_sep_newdata_dpo.sh](./scripts/llama8b/sep/instfuse_sep_newdata_dpo.sh)

## Mistral-7B-Instruct-v0.3
- All training scripts are in the [./scripts/mistral7b/sep](./scripts/mistral7b/sep) folder

---

# Evaluation

- Update OpenAI configuration file in [./datasets/openai_configs_example.yaml](./datasets/openai_configs_example.yaml), and rename it as ``/datasets/openai_configs.yaml``
- Taking Llama evaluation as an example.

## SEP score evaluation
1. Run [./scripts/evaluation/llama8b/sep.sh](./scripts/evaluation/llama8b/sep.sh). 
2. Once prompted, enter the CUDA device ID (single device ID), and the trained model path, e.g. ``meta-llama/Meta-Llama-3-8B-Instruct-TextTextText-instfuse-sep-none-newdata-dpo``. 
3. Run [./testing/sep/evaluation_main.py](./testing/sep/evaluation_main.py) to print the SEP scores.

## ASR evaluation
1. **Alpaca heuristic-based attacks**: 
   1. Run [./scripts/evaluation/llama8b/alpaca_injection.sh](./scripts/evaluation/llama8b/alpaca_injection.sh). 
   2. Once prompted, enter the CUDA device ID (single device ID), and the trained model path, e.g. ``meta-llama/Meta-Llama-3-8B-Instruct-TextTextText-instfuse-sep-none-newdata-dpo``. 
   3. Run [./testing/evaluation_main.py](./testing/evaluation_main.py) to print the ASR under Naive, Ignore, Completion, Escape, HackaPrompt attacks.
   4. Visualize the ASR Bar Chart [./testing/asr_plot.py](./testing/asr_plot.py)

2. **Alpaca GCG-based attacks**: 
   1. GCG depends on older version of transformers, therefore, we need to setup an environment called GCG 
   ```bash
    conda create -n gcg python=3.11
    conda activate gcg
    pip install -r legacy_modeling/legacy_requirements.txt
   ```
   2. Run [./scripts/evaluation/llama8b/gcg_injection.sh](./scripts/evaluation/llama8b/gcg_injection.sh). 
   3. Once prompted, enter the CUDA device ID (single device ID), and the trained model path, e.g. ``meta-llama/Meta-Llama-3-8B-Instruct-TextTextText-instfuse-sep-none-newdata-dpo``.

3. **InjecAgent**: 
   1. Run [./scripts/evaluation/llama8b/injecagent.sh](./scripts/evaluation/llama8b/injecagent.sh). 
   2. Once prompted, enter the CUDA device ID (single device ID), and the trained model path, e.g. ``meta-llama/Meta-Llama-3-8B-Instruct-TextTextText-instfuse-sep-none-newdata-dpo``.

## Utility evaluation
1. **AlpacaEval-2.0** (** Note that this can cost up to USD 50 **): 
   1. Run [./scripts/evaluation/llama8b/alpaca_utility.sh](./scripts/evaluation/llama8b/alpaca_utility.sh). 
   2. Once prompted, enter the CUDA device ID (single device ID), and the trained model path, e.g. ``meta-llama/Meta-Llama-3-8B-Instruct-TextTextText-instfuse-sep-none-newdata-dpo``.
   3. Look for the win-rate in model-path/weighted_alpaca_eval_gpt4_turbo/leaderboard.csv
   
2. **IFEval**:
   1. Run [./scripts/evaluation/llama8b/ifeval.sh](./scripts/evaluation/llama8b/ifeval.sh). 
   2. Once prompted, enter the CUDA device ID (single device ID), and the trained model path, e.g. ``meta-llama/Meta-Llama-3-8B-Instruct-TextTextText-instfuse-sep-none-newdata-dpo``.
   3. Run [./testing/ifeval/evaluation_main.py](./testing/ifeval/evaluation_main.py) and look for ASR strict

3. **MT-Bench**:
   1. Run [./scripts/evaluation/llama8b/mtbench.sh](./scripts/evaluation/llama8b/mtbench.sh). 
   2. Once prompted, enter the CUDA device ID (single device ID), and the trained model path, e.g. ``meta-llama/Meta-Llama-3-8B-Instruct-TextTextText-instfuse-sep-none-newdata-dpo``.
   3. Run [./testing/mt_bench/gen_judgment.py](./testing/mt_bench/gen_judgment.py) with flags ``--model-path [model-path] --model-id [model name, e.g. Ours]``
   4. Visualize the Radar Chart in [./testing/mt_bench/plot.py](./testing/mt_bench/plot.py)




