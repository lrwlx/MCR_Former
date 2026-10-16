#!/bin/bash
#SBATCH --job-name=mcr5seed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=five_seed_%j.log
#
# Five-seed repeat of the flagship protocol.
#
# EDIT the two lines marked EDIT for your cluster, then:  sbatch slurm_five_seed.sh
# Optional faster alternative: submit each seed as its own array task, e.g.
#   #SBATCH --array=0-4
#   SEEDS=(42 43 44 45 46); python train_flagship.py --seed ${SEEDS[$SLURM_ARRAY_TASK_ID]}
# and run run_five_seed.py afterwards with --keep-going so it only aggregates.

set -euo pipefail

module load python/3.11        # EDIT: module name on your cluster
source activate mcrformer      # EDIT: conda env, or use: source venv/bin/activate

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

python train_flagship.py --seed 42 --data composite_v13_dataset.csv --epochs 60

python run_five_seed.py --seeds 42,43,44,45,46 --data composite_v13_dataset.csv --epochs 60
