#!/bin/bash
#SBATCH --job-name=mcr_persize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=per_size_%j.log
#
# Per-thickness baseline sweep.
#
# EDIT the module/conda lines for your cluster, then:  sbatch slurm_per_size.sh
# To spread the thicknesses over several jobs instead, add
#   #SBATCH --array=0-17
# and replace --layers below with the single thickness for this task, e.g.
#   LAYERS=(3 4 5 ... 20); python per_size_baseline.py --layers ${LAYERS[$SLURM_ARRAY_TASK_ID]} ...
# (write to per-task --out-dir folders and concatenate the CSV files afterwards).

set -euo pipefail

module load python/3.11        # EDIT: module name on your cluster
source activate mcrformer      # EDIT: conda env, or use: source venv/bin/activate

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# threshold mode: how many samples does each per-thickness model need for R2 >= 0.95
python per_size_baseline.py \
  --layers 3-20 \
  --budgets 25,50,100,150,200,300,400 \
  --seeds 3 \
  --threshold 0.95 \
  --epochs 300

# after the sweep, per_size_summary.json holds the two numbers the manuscript needs:
#   samples_needed_sum        total training samples the per-size approach needs
#   unified_model_budget      200 samples/layer x number of layers for the unified model
