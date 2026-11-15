#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
LAST=${LAST_EPOCH:-099}
for run in runs/pretrain/*/; do
  name=$(basename "$run")
  for seed in 0 1 2; do
    ck="$run/seed$seed/epoch$LAST.pt"
    [ -f "$ck" ] && python tools/probe_forecast.py --ckpt "$ck" --root "$HOI4D" --out "probe/${name}_s$seed"
  done
done
s() { echo "probe/$1_s0.npz,probe/$1_s1.npz,probe/$1_s2.npz"; }
python tools/aggregate_probe.py --out probe/horizon_probe.csv \
  --series a_target=tq:$(s block) --series a_anchor=model:$(s block) \
  --series leak_target=tq:$(s leak) --series leak_anchor=model:$(s leak) \
  --series last=model:$(s last) --series pointwm=model:$(s pointwm)
