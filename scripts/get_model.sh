#!/usr/bin/env bash
# Download the checkpoint from the latest successful `train` workflow run and
# start chatting with it. Usage: scripts/get_model.sh [run_id]
set -euo pipefail
cd "$(dirname "$0")/.."

run_id="${1:-}"
if [ -z "$run_id" ]; then
  run_id="$(gh run list --workflow=train.yml --status success --limit 1 \
    --json databaseId --jq '.[0].databaseId')"
fi
[ -n "$run_id" ] && [ "$run_id" != "null" ] || { echo "no successful train run found"; exit 1; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
echo "downloading artifact from run $run_id ..."
gh run download "$run_id" --name slm-pt-run -D "$tmp"

model="$(find "$tmp" -name model.bin -print -quit)"
[ -n "$model" ] || { echo "model.bin not found in artifact"; exit 1; }
mkdir -p runs
cp "$model" runs/model.bin
echo "checkpoint -> runs/model.bin"

if [ ! -x build/slmpt_chat ]; then
  echo "building ..."
  make -j"$(nproc)"
fi
exec ./build/slmpt_chat --model runs/model.bin --temp 0.3 --top-k 40 --top-p 0.9 --rep 1.15