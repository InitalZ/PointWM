#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  CONFIGS=(configs/pretrain/pointwm.yaml configs/pretrain/block.yaml configs/pretrain/leak.yaml
           configs/pretrain/last.yaml configs/pretrain/teacher_act.yaml configs/pretrain/ablation/*.yaml)
fi
for cfg in "${CONFIGS[@]}"; do
  for seed in 0 1 2; do
    torchrun --nproc_per_node 4 train_pretrain.py --config "$cfg" --seed "$seed" --resume auto
  done
done
