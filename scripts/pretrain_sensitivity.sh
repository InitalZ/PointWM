#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SWEEP=(
  "model.causal.num_targets=2" "model.causal.num_targets=4" "model.causal.num_targets=8"
  "model.causal.gap=2"
  "model.latent_action.act_dim=4" "model.latent_action.act_dim=16" "model.latent_action.act_dim=32"
  "model.latent_action.prior_sub_p=0.0" "model.latent_action.prior_sub_p=0.1" "model.latent_action.prior_sub_p=0.5"
  "loss.beta_p=0.1" "loss.beta_p=1.0"
  "loss.lambda_pos=1.0" "loss.lambda_pos=10.0"
  "loss.lambda_vic=0.05" "loss.lambda_vic=0.5"
)
for s in "${SWEEP[@]}"; do
  name=$(echo "$s" | sed 's/.*\.\([a-z_]*\)=\(.*\)/\1_\2/')
  for seed in 0 1 2; do
    torchrun --nproc_per_node 4 train_pretrain.py --config configs/pretrain/pointwm.yaml --seed "$seed" \
      --resume auto --set "$s" "out_dir=runs/pretrain/sens_${name}"
  done
done
