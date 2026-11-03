#!/usr/bin/env bash
# Download a bounded number of WOMD scenario shards with gsutil (requires `gcloud auth login` by a
# Google account that accepted the Waymo Open Dataset terms at https://waymo.com/open).
# Usage: scripts/download_womd.sh [N_TRAIN_SHARDS=24] [N_VAL_SHARDS=8] [VERSION=1_3_0]
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_ROOT=$(python3 -c "import yaml;print(yaml.safe_load(open('configs/paths.yaml'))['data_root'])")
GS="$DATA_ROOT/tools/google-cloud-sdk/bin/gsutil"
N_TRAIN=${1:-24}; N_VAL=${2:-8}; VER=${3:-1_3_0}
BUCKET="gs://waymo_open_dataset_motion_v_${VER}/uncompressed/scenario"
RAW="$DATA_ROOT/raw/womd_v${VER}/scenario"
mkdir -p "$RAW/training" "$RAW/validation" manifests
echo "listing $BUCKET ..."
$GS ls "$BUCKET/" | tee manifests/bucket_listing_v${VER}.txt
# record version + listing checksum
python3 - "$VER" <<'EOF'
import json,sys,hashlib,time
ver=sys.argv[1]
txt=open(f"manifests/bucket_listing_v{ver}.txt").read()
json.dump({"dataset":"Waymo Open Motion Dataset","version":ver.replace('_','.'),"raw_subdir":f"womd_v{ver}","format":"scenario tfrecord",
           "bucket":f"gs://waymo_open_dataset_motion_v_{ver}","listed":time.strftime('%Y-%m-%d'),"listing_sha256":hashlib.sha256(txt.encode()).hexdigest()},
          open("manifests/dataset_version.json","w"),indent=1)
EOF
free_floor=$(python3 -c "import yaml;print(yaml.safe_load(open('configs/paths.yaml'))['min_free_bytes'])")
dl() { # split, index, total
  local f="$1.tfrecord-$(printf %05d $2)-of-$(printf %05d $3)"
  local dst="$RAW/$1/$f"
  if [ -s "$dst" ]; then echo "have $f"; return; fi
  local avail=$(df --output=avail -B1 "$DATA_ROOT" | tail -1)
  if [ "$avail" -lt "$free_floor" ]; then echo "FREE-SPACE FLOOR REACHED; stopping"; exit 3; fi
  $GS -q cp "$BUCKET/$1/$f" "$dst"
  echo "got $f ($(du -h "$dst" | cut -f1))"
}
for i in $(seq 0 $((N_TRAIN-1))); do dl training $i 1000; done
for i in $(seq 0 $((N_VAL-1))); do dl validation $i 150; done
# checksums (md5 from gsutil listing is in bucket; record local sha256 of first MiB per shard for speed)
python3 - "$RAW" <<'EOF'
import sys,hashlib,json,os
raw=sys.argv[1]; out={}
for sp in ("training","validation"):
    for f in sorted(os.listdir(os.path.join(raw,sp))):
        p=os.path.join(raw,sp,f); out[f"{sp}/{f}"]={"bytes":os.path.getsize(p),"sha256_first_mib":hashlib.sha256(open(p,'rb').read(1<<20)).hexdigest()}
json.dump(out,open("manifests/raw_shards.json","w"),indent=1); print(f"recorded {len(out)} shards")
EOF
du -sh "$RAW"
