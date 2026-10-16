#!/usr/bin/env bash
# One-shot runner for both revision experiments on a Linux server (H100 node).
#
#   cd <folder containing this script>
#   USE_CUDA_TORCH=1 JOBS=8 bash run_all_h100.sh
#
# Environment variables: USE_CUDA_TORCH=1 (CUDA PyTorch), JOBS=<n> parallel per-size jobs,
#                        ENVDIR=<path> virtualenv location (default ~/.venvs/mcrformer)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY=${PY:-python3}
ENVDIR=${ENVDIR:-$HOME/.venvs/mcrformer}
JOBS=${JOBS:-8}

echo "MCR-Former revision experiments"
echo "  working dir : $ROOT"
echo "  python      : $($PY --version 2>&1)"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "  gpu         : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
  echo "  gpu         : none detected (CPU is enough for this model)"
fi

echo "[1/5] environment"
[ -d "$ENVDIR" ] || "$PY" -m venv "$ENVDIR"
source "$ENVDIR/bin/activate"
python -m pip install --quiet --upgrade pip
if [ "${USE_CUDA_TORCH:-0}" = "1" ]; then
  python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124
else
  python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
fi
python -m pip install --quiet -r requirements.txt
python -c "import torch;print('  torch',torch.__version__,'| cuda',torch.cuda.is_available())"

echo "[2/5] smoke tests"
cd "$ROOT/experiments/per_size_baseline"
python per_size_baseline.py --layers 16 --budgets 25 --seeds 1 --epochs 3 --out-dir _smoke >/dev/null
echo "  per-size smoke test OK"
cd "$ROOT/experiments/v13_main/five_seed"
timeout 900 python train_flagship.py --seed 42 --epochs 1 --data composite_v13_dataset.csv >/dev/null
echo "  flagship smoke test OK"
cd "$ROOT/experiments/sphere_benchmark"
python sphere_benchmark.py --selftest >/dev/null
echo "  sphere benchmark self-test OK"

echo "[3/5] starting the five-seed flagship sweep in the background"
cd "$ROOT/experiments/v13_main/five_seed"
nohup python run_five_seed.py --seeds 42,43,44,45,46 > five_seed.log 2>&1 &
echo $! > five_seed.pid

echo "[4/5] starting the per-thickness sweep in the background"
cd "$ROOT/experiments/per_size_baseline"
JOBS="$JOBS" nohup bash run_per_size_parallel.sh > per_size.log 2>&1 &
echo $! > per_size.pid

echo "[5/5] running. Progress:"
echo "  tail -f $ROOT/experiments/v13_main/five_seed/five_seed.log"
echo "  tail -f $ROOT/experiments/per_size_baseline/per_size.log"
echo "Send back when finished:"
echo "  $ROOT/experiments/v13_main/five_seed/five_seed_summary.json"
echo "  $ROOT/experiments/per_size_baseline/per_size_summary.json"
