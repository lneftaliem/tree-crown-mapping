#!/bin/bash
# Reproduce the rasters supplied by Christian Igel for a few archive tiles (one tile per array task).
#SBATCH --job-name=si_test
#SBATCH --partition=serc
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=1-5
#SBATCH --output=%x_%A_%a.out
set -uo pipefail
source $HOME/split_workflow/config.sh
OUT=$WORK/tests/center_based_relabeled; CENTERS=$WORK/tests/centers; mkdir -p $OUT $CENTERS
export OUT CENTERS
ml purge; ml python/3.12.1; source $VENV/bin/activate
NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $HOME/split_workflow/test_tiles.txt)
/usr/bin/time -v bash $HOME/split_workflow/run_tile.sh "$NAME" zip 2>&1 | grep -E "DONE|SKIP|CHECK|Error|error|Traceback|Elapsed|Maximum resident"
