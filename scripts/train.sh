#!/bin/bash
# Launch V-JEPA pretraining on the nuScenes-mini clips (single GPU).
# Run from the repo root after cloning facebookresearch/jepa into ./jepa (see README).
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/logs/vits16_nuscenes"
cd "$ROOT/jepa"
python -m app.main --fname "$ROOT/configs/vits16_nuscenes.yaml" --devices cuda:0
