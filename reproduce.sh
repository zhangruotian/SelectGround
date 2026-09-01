#!/usr/bin/env bash
set -euo pipefail
DATA=${1:?Usage: bash reproduce.sh DATA_DIRECTORY OUTPUT_DIRECTORY}
OUT=${2:?Usage: bash reproduce.sh DATA_DIRECTORY OUTPUT_DIRECTORY}
export PYTHONHASHSEED=20260625
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --multi_gpu --num_machines 1 --dynamo_backend no --mixed_precision bf16 --num_processes 2 --main_process_port 0 train.py \
  --model 8b --data "$DATA" --gpus 2 --accumulation 64 \
  --pairs-file "$DATA/data/all_pairs.jsonl" --replay-file "$DATA/data/all_replay.jsonl" \
  --stage main --steps 100 --learning-rate 3.45e-5 --scheduler-steps 320 \
  --warmup-steps 10 --selector-learning-rate 1e-4 --aux-weight .1 \
  --margin .3 --pair-weight .5 --pair-every 2 --save-every 10 \
  --seed 20260625 --output "$OUT/main-100"
accelerate launch --multi_gpu --num_machines 1 --dynamo_backend no --mixed_precision bf16 --num_processes 2 --main_process_port 0 train.py \
  --model 8b --data "$DATA" --gpus 2 --accumulation 64 \
  --pairs-file "$DATA/data/all_pairs.jsonl" --replay-file "$DATA/data/all_replay.jsonl" \
  --stage main --steps 110 --learning-rate 3.45e-5 --scheduler-steps 320 \
  --warmup-steps 10 --selector-learning-rate 1e-4 --aux-weight .1 \
  --margin .3 --pair-weight .5 --pair-every 2 --save-every 10 \
  --seed 20260625 --checkpoint "$OUT/main-100" --output "$OUT/main-110"
accelerate launch --multi_gpu --num_machines 1 --dynamo_backend no --mixed_precision bf16 --num_processes 2 --main_process_port 0 train.py \
  --model 8b --data "$DATA" --gpus 2 --accumulation 64 \
  --stage refinement --steps 5 --learning-rate 1e-6 --scheduler-steps 30 \
  --warmup-steps 10 --selector-learning-rate 1e-4 --aux-weight .05 \
  --margin .3 --pair-weight .5 --pair-every 3 --save-every 5 \
  --seed 20260625 --initialize-from "$OUT/main-110" --output "$OUT/refinement-5"
accelerate launch --multi_gpu --num_machines 1 --dynamo_backend no --mixed_precision bf16 --num_processes 2 --main_process_port 0 train.py \
  --model 8b --data "$DATA" --gpus 2 --accumulation 64 \
  --stage refinement --steps 10 --learning-rate 1e-6 --scheduler-steps 30 \
  --warmup-steps 10 --selector-learning-rate 1e-4 --aux-weight .05 \
  --margin .3 --pair-weight .5 --pair-every 3 --save-every 5 \
  --seed 20260625 --checkpoint "$OUT/refinement-5" --output "$OUT/final"
