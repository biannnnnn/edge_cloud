#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="统计目录/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/run_${TS}.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

python3 glm_evaluate2.py \
  --model deepseek \
  --num_games 50 \
  --backend openai \
  --temperature 0.7 \
  --max_tokens 2000 \
  --log_file "$LOG_FILE" \
  "$@"

echo "评估完成，日志: $LOG_FILE"
