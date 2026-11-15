#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PRE=$1; NAME=${2:-$(basename "$PRE")}; LAST=${LAST_EPOCH:-099}
for frac in 0.01 0.05 0.1 0.2 0.5 1.0; do
  for seed in 0 1 2; do
    if [ "$PRE" = none ]; then CK=none; else CK="$PRE/seed$seed/epoch$LAST.pt"; fi
    torchrun --nproc_per_node 4 finetune_hoi4d_seg.py --config configs/finetune/hoi4d_offline.yaml \
      --seed "$seed" --init_ckpt "$CK" --out_dir "runs/label_eff/$NAME/frac$frac/seed$seed" --set data.label_frac=$frac
  done
done
