#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/home/liuheng/edge_cloude/fixed_ratio_experiments}"
LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-/home/liuheng/LLaMA-Factory}"
PYTHON_BIN="${PYTHON_BIN:-/home/liuheng/miniconda3/envs/llama_factory/bin/python}"
GPU_ID="${GPU_ID:-0}"
NUM_GAMES="${NUM_GAMES:-50}"
SOURCE_LIMIT="${SOURCE_LIMIT:-584}"
GENERATION_RETRIES="${GENERATION_RETRIES:-5}"
DEEPSEEK_ENV_FILE="${DEEPSEEK_ENV_FILE:-/home/liuheng/.config/edge_cloud/deepseek.env}"

wait_for_gpu() {
  local gpu_id="$1"
  local used
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id")
    if [[ "$used" -le 2000 ]]; then
      echo "GPU $gpu_id is ready (${used} MiB used)"
      return
    fi
    echo "waiting for GPU $gpu_id (${used} MiB used)"
    sleep 60
  done
}

if [[ -z "${DEEPSEEK_API_KEY:-}" && -f "$DEEPSEEK_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$DEEPSEEK_ENV_FILE"
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is not set in the environment or $DEEPSEEK_ENV_FILE" >&2
  exit 2
fi

mkdir -p "$EXPERIMENT_ROOT/logs"
cd "$EXPERIMENT_ROOT"

"$PYTHON_BIN" generate_fixed_ratio_cot.py \
  --model deepseek-v4-pro \
  --ratios 0.2 0.4 0.6 0.8 \
  --limit "$SOURCE_LIMIT" \
  --retries "$GENERATION_RETRIES" \
  2>&1 | tee logs/generate.log

"$PYTHON_BIN" validate_fixed_ratio.py --data_dir data --expected_records "$SOURCE_LIMIT" \
  2>&1 | tee logs/validate.log

"$PYTHON_BIN" prepare_fixed_ratio_sft.py \
  2>&1 | tee logs/prepare.log

cd "$LLAMA_FACTORY_ROOT"
wait_for_gpu 0
wait_for_gpu 1
GPU_ID=0 PERCENTAGES="020 060" bash train_fixed_ratios.sh &
train_gpu0_pid=$!
GPU_ID=1 PERCENTAGES="040 080" bash train_fixed_ratios.sh &
train_gpu1_pid=$!
wait "$train_gpu0_pid"
wait "$train_gpu1_pid"
GPU_ID="$GPU_ID" NUM_GAMES="$NUM_GAMES" bash evaluate_fixed_ratios.sh
"$PYTHON_BIN" "$EXPERIMENT_ROOT/collect_fixed_ratio_results.py"
