#!/usr/bin/env bash
set -euo pipefail
DATA=${1:?Usage: bash reproduce.sh DATA_DIRECTORY OUTPUT_DIRECTORY}
OUT=${2:?Usage: bash reproduce.sh DATA_DIRECTORY OUTPUT_DIRECTORY}
export PYTHONHASHSEED=20260625
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --multi_gpu --num_machines 1 --dynamo_backend no --mixed_precision bf16 --num_processes 2 train.py \
  --data "$DATA" --stage main --steps 100 --output "$OUT/main-100"
accelerate launch --multi_gpu --num_machines 1 --dynamo_backend no --mixed_precision bf16 --num_processes 2 train.py \
  --data "$DATA" --stage main --steps 110 --resume "$OUT/main-100" --output "$OUT/main-110"
accelerate launch --multi_gpu --num_machines 1 --dynamo_backend no --mixed_precision bf16 --num_processes 2 train.py \
  --data "$DATA" --stage refinement --steps 5 --initialize-from "$OUT/main-110" --output "$OUT/refinement-5"
accelerate launch --multi_gpu --num_machines 1 --dynamo_backend no --mixed_precision bf16 --num_processes 2 train.py \
  --data "$DATA" --stage refinement --steps 10 --resume "$OUT/refinement-5" --output "$OUT/final"
