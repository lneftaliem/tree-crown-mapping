#!/bin/bash
# Re-plot Supplementary Figs S1-S3 (final pipeline, tau = 0.7) with the figure-format copy of the evaluator
# (evaluate_object_accuracy_figfix.py: identical computations; tight bounding box, S2 canvas, S3 legend).
# Inputs are the cached predictions and instance labels of job 50693038's threshold run; no new inference.
# Writes only $GROUP_SCRATCH/leonan_split_2026/threshold_v1/final_0.7_figfix/.
#SBATCH --job-name=val_figfix
#SBATCH --partition=serc
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
V=$GROUP_SCRATCH/leonan_split_2026/threshold_v1
EVAL=$HOME/unet_workflow/validation/evaluate_object_accuracy_figfix.py
META=$HOME/unet_workflow/validation/tile_metadata.csv
PRE=/home/groups/cfield/myenv/notebooks/Preprocessing
MD=$GROUP_SCRATCH/myenv_scratch/notebooks_scratch/saved_models/UNet
MODEL=$MD/trees_20260120-1928_AdaDelta_weightmap_tversky_012_256_final.keras
cd $V
ml purge; ml py-tensorflow/2.18.0_py312 viz py-pillow/10.2.0_py312 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=-1
python3 $EVAL --preproc_dir $PRE --tile_metadata_path $META --model_dir $MD --model_path $MODEL --connectivity 1 --min_crown_area 1 \
  --input_channels ndvi_pan --output_dir $V/final_0.7_figfix --cached_predictions $V/cc_0.7/predictions.npy --pred_threshold 0.7 \
  --instance_labels_npy $V/instances_0.7.npy --min_crown_m2 3.0 > final_0.7_figfix.log 2>&1
echo "final_0.7_figfix exit $?"
