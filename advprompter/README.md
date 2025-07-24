# Training

## StruQ
```bash
python -m advprompter.main --config-name test target_llm=spcl_delm_llm \
target_llm.llm_params.model_name=Llama-3.2-3b-SpclSpclSpcl-struq \
target_llm.llm_params.checkpoint=meta-llama/Llama-3.2-3b-SpclSpclSpcl-struq-sep-none \
train.prompter_optim_params.lr=1e-5 train.dataset_pth=./advprompter/data/prompt_injections/dataset/test_SpclSpclSpcl_datasets_davinci_003_outputs_json.csv eval.data.dataset_pth_dct.train=./advprompter/data/prompt_injections/dataset/test_SpclSpclSpcl_datasets_davinci_003_outputs_json.csv wandb_params.enable_wandb=false
```

## SecAlign
```bash
python -m advprompter.main --config-name test target_llm=spcl_delm_llm \
target_llm.llm_params.model_name=Llama-3.2-3b_SpclSpclSpcl-secalign \
target_llm.llm_params.checkpoint=meta-llama/Llama-3.2-3b_SpclSpclSpcl-secalign \
train.prompter_optim_params.lr=1e-5 train.dataset_pth=./advprompter/data/prompt_injections/dataset/test_SpclSpclSpcl_datasets_davinci_003_outputs_json.csv eval.data.dataset_pth_dct.train=./advprompter/data/prompt_injections/dataset/test_SpclSpclSpcl_datasets_davinci_003_outputs_json.csv wandb_params.enable_wandb=false
```


## ISE
```bash
python -m advprompter.main --config-name test target_llm=spcl_delm_llm \
target_llm.llm_params.model_name=Llama-3.2-3b-SpclSpclSpcl-ise \
target_llm.llm_params.checkpoint=meta-llama/Llama-3.2-3b-SpclSpclSpcl-ise \
target_llm.llm_params.customized_model_class=LlamaForCausalLMMoE \
target_llm.llm_params.pass_expert_labels=true \
train.prompter_optim_params.lr=1e-5 train.dataset_pth=./advprompter/data/prompt_injections/dataset/test_SpclSpclSpcl_datasets_davinci_003_outputs_json.csv eval.data.dataset_pth_dct.train=./advprompter/data/prompt_injections/dataset/test_SpclSpclSpcl_datasets_davinci_003_outputs_json.csv wandb_params.enable_wandb=false
```

## Ours
```bash
python -m advprompter.main --config-name test target_llm=spcl_delm_llm \
target_llm.llm_params.model_name=Llama-3.2-3b-SpclSpclSpcl-instfuse \
target_llm.llm_params.checkpoint=meta-llama/Llama-3.2-3b-SpclSpclSpcl-instfuse \
target_llm.llm_params.customized_model_class=LlamaForCausalLMFuse \
target_llm.llm_params.pass_expert_labels=true \
train.prompter_optim_params.lr=1e-5 train.dataset_pth=./advprompter/data/prompt_injections/dataset/test_SpclSpclSpcl_datasets_davinci_003_outputs_json.csv eval.data.dataset_pth_dct.train=./advprompter/data/prompt_injections/dataset/test_SpclSpclSpcl_datasets_davinci_003_outputs_json.csv wandb_params.enable_wandb=false
```


# Testing

