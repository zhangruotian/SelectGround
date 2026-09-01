#!/usr/bin/env bash
set -euo pipefail
DATA=${1:?Usage: bash reproduce.sh DATA_DIRECTORY OUTPUT_DIRECTORY}
OUT=${2:?Usage: bash reproduce.sh DATA_DIRECTORY OUTPUT_DIRECTORY}
export PYTHONHASHSEED=20260625
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --multi_gpu --num_machines 1 --dynamo_backend no --mixed_precision bf16 --num_processes 2 --main_process_port 0 train.py \
  --data "$DATA" --output "$OUT"
