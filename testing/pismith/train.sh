#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate prompt  # Replace with your actual env name

# ── Usage ───────────────────────────────────────────────────────────────────
if [[ $# -lt 2 ]]; then
    cat <<EOF
Usage: $0 <model_path> <dataset> [cuda_devices]

Arguments:
  model_path     Target model path. Target type is auto-detected:
                   - contains "drip" or "instfuse" → DRIP
                   - contains "secalign"           → Meta-SecAlign
                   - otherwise                     → Llama (undefended)
  dataset        Either "alpaca" or "sep". Selects training script and output dir.
  cuda_devices   Optional. Comma-separated GPU indices. Default: "0".
                   Single GPU → uses python.
                   Multiple GPUs → uses torchrun with --nproc_per_node.

Examples:
  $0 meta-llama/Llama-3.1-8B-Instruct-TextTextText-instfuse-alpaca-dpo sep 1,2,3
  $0 meta-llama/Llama-3.1-8B-Instruct sep 4
  $0 meta-llama/Meta-SecAlign-8B-merged sep 0,2,3
EOF
    exit 1
fi

MODEL_PATH="$1"
DATASET="$2"
CUDA_DEVICES="${3:-0}"

# ── Validate dataset ────────────────────────────────────────────────────────
case "$DATASET" in
    alpaca)
        train_module="testing.pismith.train_alpaca"
        ;;
    sep)
        train_module="testing.pismith.train_sep"
        ;;
    *)
        echo "Error: dataset must be 'alpaca' or 'sep', got '$DATASET'"
        exit 1
        ;;
esac

# ── Auto-detect target type ──────────────────────────────────────────────────
model_lower="${MODEL_PATH,,}"

if [[ "$model_lower" == *"drip"* ]] || [[ "$model_lower" == *"instfuse"* ]]; then
    target_name="DRIP"
    target_tag="drip"
    customized_class="LlamaForCausalLMDRIP"
elif [[ "$model_lower" == *"secalign"* ]]; then
    target_name="Meta-SecAlign"
    target_tag="metasecalign"
    customized_class=""
else
    target_name="Llama (undefended)"
    target_tag="llama"
    customized_class=""
fi

# ── Output directory ────────────────────────────────────────────────────────
output_dir="./pismith_ckpt/${DATASET}_${target_tag}"

# ── DDP / single-GPU detection ──────────────────────────────────────────────
n_gpus=$(echo "$CUDA_DEVICES" | tr ',' '\n' | wc -l)

# ── Report ──────────────────────────────────────────────────────────────────
echo "──────────────────────────────────────────────────────────────"
echo " Dataset           : $DATASET"
echo " Train module      : $train_module"
echo " Target detected   : $target_name"
echo " Model path        : $MODEL_PATH"
echo " Output dir        : $output_dir"
echo " Customized class  : ${customized_class:-<none>}"
echo " CUDA devices      : $CUDA_DEVICES (n=$n_gpus)"
echo " Launcher          : $([[ $n_gpus -gt 1 ]] && echo 'torchrun (DDP)' || echo 'python (single GPU)')"
echo "──────────────────────────────────────────────────────────────"

# ── Common training args ────────────────────────────────────────────────────
COMMON_ARGS=(
    --model_name_or_path "$MODEL_PATH"
    --attack_model_name "Qwen/Qwen3-4B-Instruct-2507"
    --attack_model_path "${HF_HOME}/Qwen/Qwen3-4B-Instruct-2507"
    --output_dir "$output_dir"
    --max_train_samples 100
    --num_epochs 10
    --group_size 6
    --lora_r 8
    --lora_alpha 16
    --lora_target_modules q_proj v_proj k_proj o_proj
    --resume_from_checkpoint
)

if [[ -n "$customized_class" ]]; then
    COMMON_ARGS+=(--customized_model_class "$customized_class")
fi

# ── Launch ──────────────────────────────────────────────────────────────────
if [[ $n_gpus -gt 1 ]]; then
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
    torchrun --nproc_per_node=$n_gpus -m "$train_module" "${COMMON_ARGS[@]}"
else
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
    python -m "$train_module" "${COMMON_ARGS[@]}"
fi