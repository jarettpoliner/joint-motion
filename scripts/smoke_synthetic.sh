#!/usr/bin/env bash
# SYNTHETIC smoke test of the orchestration: fixture -> preprocess -> manifests -> train (+resume) -> eval.
# Writes under <data_root>/processed_synthetic and manifests/synthetic_*.json. Not a real-data result.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-$(python3 -c "import yaml;print(yaml.safe_load(open('configs/paths.yaml'))['train_python'])")}
DATA_ROOT=$(python3 -c "import yaml;print(yaml.safe_load(open('configs/paths.yaml'))['data_root'])")
OUT=$DATA_ROOT/processed_synthetic
$PY - <<EOF
from pathlib import Path
import json, hashlib
from waymo_av.data.synthetic import write_synthetic_shard
from waymo_av.data.preprocess import process_shard
out=Path("$OUT"); out.mkdir(parents=True, exist_ok=True)
shard=write_synthetic_shard(out/"synthetic.tfrecord", n=48, seed=0)
st=process_shard(shard, out/"training")
ids=sorted(s["scenario_id"] for s in st["scenarios"])
for name, sel in [("synthetic_train", ids[:32]), ("synthetic_dev", ids[32:48])]:
    json.dump({"name":name,"processed_dir":str(out),"split_dir":"training","n":len(sel),"scenario_ids":sel,"note":"SYNTHETIC FIXTURE","sha256_ids":hashlib.sha256("\n".join(sel).encode()).hexdigest()}, open(f"manifests/{name}.json","w"), indent=1)
print("synthetic manifests written", len(ids))
EOF
$PY -m waymo_av.train --config configs/train/m0_overfit.yaml --run-name synthetic_smoke \
  --override train_manifest=manifests/synthetic_train.json dev_manifest=manifests/synthetic_dev.json total_steps=60 eval_every=30 log_every=10 dev_eval_scenes=16 num_workers=2 planned_gpu_hours=0.01
$PY -m waymo_av.train --config configs/train/m0_overfit.yaml --run-name synthetic_smoke --resume \
  --override train_manifest=manifests/synthetic_train.json dev_manifest=manifests/synthetic_dev.json total_steps=90 eval_every=30 log_every=10 dev_eval_scenes=16 num_workers=2 planned_gpu_hours=0.01
$PY -m waymo_av.evaluate --methods cp cv oracle_log oracle_tokens ckpt:$DATA_ROOT/runs/synthetic_smoke/ckpt_last.pt@synthetic_smoke \
  --conditions clean missing_0.3 noise_0.3 --manifest manifests/synthetic_dev.json --out $DATA_ROOT/runs/synthetic_smoke/eval --offroad-sign -1 --n-samples 6 --audit-samples 32 --batch-size 8
echo "SMOKE OK"
