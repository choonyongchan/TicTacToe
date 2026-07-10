#!/usr/bin/env bash
set -euo pipefail

out=./benchmark/4x4k3
mkdir -p ./benchmark
: > "$out"

agents=(
  random  
  minimax
  minimax_ab
  minimax_rewards_ab
  negamax
  mtdf
  mtdf_id
  negascout
  bns
  bns_id
)

for agent in "${agents[@]}"; do
  echo "Running agent: $agent"
  echo "=== $agent ===" >> "$out"
  python main.py -n 4 -k 3 -agt "$agent" >> "$out"
  echo >> "$out"
done