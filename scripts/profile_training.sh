#!/bin/bash
# =============================================================================
# profile_training.sh
#
# Wrapper around train_unified.py that measures:
#   - Peak active GPU memory per rank (torch.cuda.max_memory_allocated)
#   - Reserved GPU memory per rank (torch.cuda.max_memory_reserved)
#   - Wall-clock time for the full training run
#   - Per-step throughput (samples/sec, tokens/sec)
#
# Usage:
#   bash profile_training.sh llama      # profile LLaMA-3-8B
#   bash profile_training.sh mistral    # profile Mistral-7B
#   bash profile_training.sh both       # profile both sequentially
#
# Output:
#   profile_results/
#     <model>_memory_rank<N>.txt        -- peak mem per GPU
#     <model>_throughput.txt            -- timing + samples/tokens per sec
#     <model>_summary.txt               -- human-readable summary
# =============================================================================

set -euo pipefail

TARGET="${1:-both}"
SCRIPT_PATH="train_unified.py"
PROFILE_DIR="profile_results"
mkdir -p "$PROFILE_DIR"

# ---------------------------------------------------------------------------
# Shared training flags (match your actual scripts exactly)
# ---------------------------------------------------------------------------
COMMON_ARGS=(
  --bf16 True
  --per_device_train_batch_size 4
  --per_device_eval_batch_size 1
  --gradient_accumulation_steps 8
  --num_train_epochs 1
  --save_strategy "no"           # disable saving during profiling
  --logging_steps 1
  --tf32 True
  --model_max_length 512
  --dataloader_num_workers 4
  --fsdp "full_shard auto_wrap"
  --warmup_ratio 0.03
  --weight_decay 0.
  --lr_scheduler_type "cosine"
)

NPROC=6
MASTER_PORT=29951

# ---------------------------------------------------------------------------
# Model-specific config
# ---------------------------------------------------------------------------
LLAMA_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
LLAMA_DATA="datasets/sep/sep_data_cleaned_dpo_gpt.json"
LLAMA_FSDP="training/config/fsdp_config.json"
LLAMA_DELIM="TextTextText"
LLAMA_LR="1e-4"

MISTRAL_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
MISTRAL_DATA="datasets/sep/sep_data_cleaned_dpo_gpt.json"
MISTRAL_FSDP="training/config/fsdp_config_mistral.json"
MISTRAL_DELIM="TextTextTextMistral"
MISTRAL_LR="5e-5"

# ---------------------------------------------------------------------------
# Profile launcher: injects profiling wrapper around train_unified.py
# ---------------------------------------------------------------------------
run_profile() {
    local MODEL_FAMILY="$1"   # llama | mistral
    local BASE_MODEL="$2"
    local DATA_PATH="$3"
    local FSDP_CONFIG="$4"
    local DELIMITER="$5"
    local LR="$6"
    local LABEL="$7"          # human-readable label for output files

    local OUTPUT_DIR="profile_results/${LABEL}_dummy_run"
    local TIMING_FILE="${PROFILE_DIR}/${LABEL}_throughput.txt"
    local SUMMARY_FILE="${PROFILE_DIR}/${LABEL}_summary.txt"

    echo ""
    echo "================================================================="
    echo " Profiling: ${LABEL}"
    echo " Model:     ${BASE_MODEL}"
    echo " GPUs:      ${NPROC}x  (nproc_per_node=${NPROC})"
    echo " Eff. batch: $((4 * 8 * NPROC))  (per_device=4, grad_accum=8)"
    echo "================================================================="

    # Record wall-clock start
    START_TS=$(date +%s%N)   # nanoseconds


    PROFILE_DIR="$PROFILE_DIR" \
    PROFILE_LABEL="$LABEL" \
    python -m torch.distributed.run \
        --nproc_per_node="${NPROC}" \
        --master_port="${MASTER_PORT}" \
        profile_hook.py \
            --wrapped_script "${SCRIPT_PATH}" \
            --objective "dpo" \
            --model-family "${MODEL_FAMILY}" \
            --arch "fuse" \
            --model_name_or_path "${BASE_MODEL}" \
            --data_path "${DATA_PATH}" \
            --output_dir "${OUTPUT_DIR}" \
            --learning_rate "${LR}" \
            --attack "${DELIMITER}_None" \
            --fsdp_config "${FSDP_CONFIG}" \
            "${COMMON_ARGS[@]}" \
    2>&1 | tee "${PROFILE_DIR}/${LABEL}_train.log"

    END_TS=$(date +%s%N)
    ELAPSED_NS=$(( END_TS - START_TS ))
    ELAPSED_SEC=$(echo "scale=2; $ELAPSED_NS / 1000000000" | bc)
    ELAPSED_HOURS=$(echo "scale=4; $ELAPSED_SEC / 3600" | bc)

    # Count samples from dataset
    TOTAL_SAMPLES=$(python3 -c "
import json
with open('${DATA_PATH}') as f:
    data = json.load(f)
print(len(data) if isinstance(data, list) else sum(len(v) for v in data.values()))
" 2>/dev/null || echo "unknown")

    # Write timing file
    {
        echo "Model:           ${BASE_MODEL}"
        echo "Dataset:         ${DATA_PATH}"
        echo "Total samples:   ${TOTAL_SAMPLES}"
        echo "Effective batch: $((4 * 8 * NPROC))"
        echo "Seq length:      512"
        echo "Num GPUs:        ${NPROC}"
        echo ""
        echo "Wall-clock time: ${ELAPSED_SEC} seconds"
        echo "Wall-clock time: ${ELAPSED_HOURS} GPU-hours (x${NPROC} GPUs = $(echo "scale=4; $ELAPSED_HOURS * $NPROC" | bc) total GPU-hours)"
        if [ "$TOTAL_SAMPLES" != "unknown" ]; then
            SAMPLES_PER_SEC=$(echo "scale=2; $TOTAL_SAMPLES / $ELAPSED_SEC" | bc 2>/dev/null || echo "N/A")
            TOKENS_PER_SEC=$(echo "scale=0; $TOTAL_SAMPLES * 512 / $ELAPSED_SEC" | bc 2>/dev/null || echo "N/A")
            echo "Throughput:      ${SAMPLES_PER_SEC} samples/sec"
            echo "Throughput:      ${TOKENS_PER_SEC} tokens/sec"
        fi
    } > "$TIMING_FILE"

    echo "Timing written to: $TIMING_FILE"

    # Aggregate memory across ranks (written by profile_hook.py)
    python3 - <<EOF >> "$SUMMARY_FILE"
import os, glob

label = "${LABEL}"
profile_dir = "${PROFILE_DIR}"

mem_files = sorted(glob.glob(f"{profile_dir}/{label}_memory_rank*.txt"))
if not mem_files:
    print(f"[{label}] No per-rank memory files found.")
else:
    alloc_vals, reserved_vals = [], []
    for f in mem_files:
        with open(f) as fh:
            lines = {l.split(":")[0].strip(): float(l.split(":")[1].strip().split()[0])
                     for l in fh if ":" in l}
        alloc_vals.append(lines.get("peak_allocated_GB", 0))
        reserved_vals.append(lines.get("peak_reserved_GB", 0))
    print(f"[{label}] Per-GPU peak allocated: min={min(alloc_vals):.2f} GB  max={max(alloc_vals):.2f} GB  mean={sum(alloc_vals)/len(alloc_vals):.2f} GB")
    print(f"[{label}] Per-GPU peak reserved:  min={min(reserved_vals):.2f} GB  max={max(reserved_vals):.2f} GB  mean={sum(reserved_vals)/len(reserved_vals):.2f} GB")

with open(f"{profile_dir}/{label}_throughput.txt") as fh:
    print(fh.read())
EOF

    echo "Summary written to: $SUMMARY_FILE"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$TARGET" in
    llama)
        run_profile "llama" "$LLAMA_MODEL" "$LLAMA_DATA" "$LLAMA_FSDP" "$LLAMA_DELIM" "$LLAMA_LR" "llama3_8b"
        ;;
    mistral)
        run_profile "mistral" "$MISTRAL_MODEL" "$MISTRAL_DATA" "$MISTRAL_FSDP" "$MISTRAL_DELIM" "$MISTRAL_LR" "mistral_7b"
        ;;
    both)
        run_profile "llama"   "$LLAMA_MODEL"   "$LLAMA_DATA"   "$LLAMA_FSDP"   "$LLAMA_DELIM"   "$LLAMA_LR"   "llama3_8b"
        run_profile "mistral" "$MISTRAL_MODEL" "$MISTRAL_DATA" "$MISTRAL_FSDP" "$MISTRAL_DELIM" "$MISTRAL_LR" "mistral_7b"
        ;;
    *)
        echo "Usage: $0 [llama|mistral|both]"
        exit 1
        ;;
esac

echo ""
echo "All profiling complete. Results in: ${PROFILE_DIR}/"
ls -lh "${PROFILE_DIR}/"