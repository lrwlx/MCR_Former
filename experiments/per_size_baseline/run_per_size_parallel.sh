#!/usr/bin/env bash
# Sweep the per-thickness baseline over all thicknesses in parallel.
# Each thickness writes to its own folder under out/; results are merged at the end.
#
#   JOBS=8 bash run_per_size_parallel.sh
#
set -euo pipefail
cd "$(dirname "$0")"

JOBS=${JOBS:-8}
BUDGETS=${BUDGETS:-25,50,100,150,200,300,400}
SEEDS=${SEEDS:-3}
FIRST=${FIRST:-3}
LAST=${LAST:-20}

mkdir -p out

echo "thicknesses ${FIRST}-${LAST} | budgets ${BUDGETS} | seeds ${SEEDS} | parallel jobs ${JOBS}"

seq "$FIRST" "$LAST" | xargs -P "$JOBS" -I{} sh -c '
  python per_size_baseline.py \
    --layers {} \
    --budgets '"$BUDGETS"' \
    --seeds '"$SEEDS"' \
    --out-dir "out/layer_{}" > "out/layer_{}.log" 2>&1
  echo "  finished thickness {}"
'

echo "merging per-thickness results"
python merge_per_size.py
