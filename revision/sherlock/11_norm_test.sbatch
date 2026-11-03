#!/bin/bash
# TEST of the inference normalization extent (per patch / per tile / per scene), NDVI+PAN, same model as v1.
# Task 1: the 50 re-extractable validation patches; tasks 2-4: 4 v1 tiles each (Fig 6 non-vegetated tiles + 2 random per city).
# Writes only $GROUP_SCRATCH/leonan_split_2026/norm_test/.
#SBATCH --job-name=norm_test
#SBATCH --partition=serc
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --array=1-4
#SBATCH --output=logs/%x_%A_%a.out
set -uo pipefail
NB=$GROUP_SCRATCH/myenv_scratch/notebooks_scratch
OUT=$GROUP_SCRATCH/leonan_split_2026/norm_test${ORDER:+_$ORDER}
export ORDER=${ORDER:-b2b1}
export NB_DIR=$NB RAW_DIR=$NB/raw_images
export MODEL=$NB/saved_models/UNet/trees_20260120-1928_AdaDelta_weightmap_tversky_012_256_final.keras
ml purge; ml math py-tensorflow/2.18.0_py312
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=-1
export PYTHONPATH=$HOME/unet_workflow:${PYTHONPATH:-}
M=$HOME/unet_workflow/norm_test
if [ "$SLURM_ARRAY_TASK_ID" = 1 ]; then
  python3 $M/norm_test.py patches $M/patches.csv $OUT || exit 4
else
  python3 $M/norm_test.py tiles $M/tiles.txt $(( (SLURM_ARRAY_TASK_ID - 2) * 4 )) 4 $OUT || exit 4
fi
