#!/usr/bin/env bash
# Compile the vendored (trimmed) WOMD scenario/map protos into src/waymo_av/protos_gen.
# Requires grpcio-tools in the active python (requirements/train.in).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
mkdir -p src/waymo_av/protos_gen
$PY -m grpc_tools.protoc -I protos \
  --python_out=src/waymo_av/protos_gen \
  protos/waymo_open_dataset/protos/map.proto \
  protos/waymo_open_dataset/protos/scenario.proto
# make the generated package importable as waymo_av.protos_gen.waymo_open_dataset.protos.*
find src/waymo_av/protos_gen -type d -exec touch {}/__init__.py \;
# generated modules import `from waymo_open_dataset.protos import map_pb2`; rewrite to package-relative.
sed -i 's/^from waymo_open_dataset\.protos import /from waymo_av.protos_gen.waymo_open_dataset.protos import /' \
  src/waymo_av/protos_gen/waymo_open_dataset/protos/*_pb2.py
echo "protos built: $(ls src/waymo_av/protos_gen/waymo_open_dataset/protos/*_pb2.py)"
