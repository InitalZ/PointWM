#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PRE=$1; NAME=${2:-$(basename "$PRE")}; LAST=${LAST_EPOCH:-099}
for seed in 0 1 2; do
  if [ "$PRE" = none ]; then CK=none; else CK="$PRE/seed$seed/epoch$LAST.pt"; fi
  torchrun --nproc_per_node 4 finetune_hoi4d_seg.py --config configs/finetune/hoi4d_offline.yaml \
    --seed "$seed" --init_ckpt "$CK" --out_dir "runs/seg_offline/$NAME/seed$seed"
  torchrun --nproc_per_node 4 finetune_hoi4d_seg.py --config configs/finetune/hoi4d_offline.yaml \
    --seed "$seed" --init_ckpt "$CK" --out_dir "runs/seg_offline_l10/$NAME/seed$seed" --set data.label_frac=0.1
  torchrun --nproc_per_node 4 finetune_hoi4d_seg.py --config configs/finetune/hoi4d_online.yaml \
    --seed "$seed" --init_ckpt "$CK" --out_dir "runs/seg_online/$NAME/seed$seed"
  torchrun --nproc_per_node 4 finetune_msr.py --config configs/finetune/msr.yaml \
    --seed "$seed" --init_ckpt "$CK" --out_dir "runs/msr/$NAME/seed$seed"
done
