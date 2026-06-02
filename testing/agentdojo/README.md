

### DRIP

Install AgentDojo
```bash
pip install agentdojo==0.1.35
```

Test on undefended model with no attack
```bash
bash run_local_vlm.sh # start local vllm server
python -m testing.agentdojo.run_agentdojo \
  --mode official \
  -m local \
  --logdir agentdojo_runs/llama8b \
```

Test on undefended model with attacks
```bash
bash run_local_vlm.sh # start local vllm server
python -m testing.agentdojo.run_agentdojo \
  --mode official \
  -m local \
  --logdir agentdojo_runs/llama8b \
  --attack [important_instructions|ignore_previous]
```

Test on Meta SecAlign model with no attack
```bash
bash run_local_vlm_metasecalign.sh # start local vllm server
python -m testing.agentdojo.run_agentdojo \
  --mode official \
  -m local \
  --logdir agentdojo_runs/metasecalign8b \
  --tool-delimiter input
```

Test on Meta SecAlign model with attacks
```bash
bash run_local_vlm_metasecalign.sh # start local vllm server
python -m testing.agentdojo.run_agentdojo \
  --mode official \
  -m local \
  --logdir agentdojo_runs/metasecalign8b \
  --tool-delimiter input \
  --attack [important_instructions|ignore_previous]
```


Test on DRIP with no attack
```bash
python -m testing.agentdojo.run_agentdojo \
  --mode fuse \
  --model_name_or_path [model_path] \
  --customized_model_class LlamaForCausalLMDRIP \
  --logdir ./agentdojo_runs/llama8b_drip \
  --attack [important_instructions|ignore_previous]
```

Test on DRIP with attacks
```bash
python -m testing.agentdojo.run_agentdojo \
  --mode fuse \
  --model_name_or_path [model_path] \
  --customized_model_class LlamaForCausalLMDRIP \
  --logdir ./agentdojo_runs/llama8b_drip \
  --attack [important_instructions|ignore_previous]
```
