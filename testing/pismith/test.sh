#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate prompt  # Replace with your actual env name

# ── Usage ───────────────────────────────────────────────────────────────────
if [[ $# -lt 2 ]]; then
    cat <<EOF
Usage: $0 <model_path> <dataset> [cuda_device]

Arguments:
  model_path    Target model path. Target type is auto-detected from path:
                  - contains "drip" or "instfuse" → DRIP
                  - contains "secalign"           → Meta-SecAlign
                  - otherwise                     → Llama (undefended)
  dataset       Either "alpaca" or "sep". Selects which test module to run
                and which PISmith attack adapter to load.
  cuda_device   Optional. GPU index, default 0.

Examples:
  $0 meta-llama/Llama-3.1-8B-Instruct-log-TextTextText-instfuse-alpaca-dpo alpaca 2
  $0 meta-llama/Meta-SecAlign-8B-merged sep 3
EOF
    exit 1
fi

MODEL_PATH="$1"
DATASET="$2"
CUDA_DEVICE="${3:-0}"

# ── Validate dataset ────────────────────────────────────────────────────────
case "$DATASET" in
    alpaca)
        test_module="testing.pismith.test_pismith_alpaca"
        ;;
    sep)
        test_module="testing.pismith.test_pismith_sep"
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

# ── Attack adapter path: combines dataset + target ──────────────────────────
attack_path="./pismith_ckpt/sep_${target_tag}/attack_lm_final"

# ── Report ──────────────────────────────────────────────────────────────────
echo "──────────────────────────────────────────────────────────────"
echo " Dataset           : $DATASET"
echo " Test module       : $test_module"
echo " Target detected   : $target_name"
echo " Model path        : $MODEL_PATH"
echo " Attack adapter    : $attack_path"
echo " Customized class  : ${customized_class:-<none>}"
echo " CUDA device       : $CUDA_DEVICE"
echo "──────────────────────────────────────────────────────────────"

# ── Sanity check: attack adapter exists ─────────────────────────────────────
if [[ ! -d "$attack_path" ]]; then
    echo "Warning: attack adapter directory not found: $attack_path"
    echo "Continuing anyway in case the test script handles this..."
fi

# ── Build & run ─────────────────────────────────────────────────────────────
CMD=(python -m "$test_module"
     -m "$MODEL_PATH"
     --attack_model_path "$attack_path")

if [[ -n "$customized_class" ]]; then
    CMD+=(--customized_model_class "$customized_class")
fi

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE "${CMD[@]}"