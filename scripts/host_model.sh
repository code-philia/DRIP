# choose a log file so you can tail it
MODEL="meta-llama/Meta-Llama-3-8B-Instruct-log"   # or an absolute local path

nohup vllm serve "$MODEL"\
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name llama-fuse \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  > vllm.log 2>&1 & disown

tail -f vllm.log
