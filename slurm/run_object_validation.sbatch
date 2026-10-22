#!/bin/bash
#SBATCH --partition=serc
#SBATCH --job-name=unet_obj_val
#SBATCH --output=obj_val_%j.out
#SBATCH --error=obj_val_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4g
#
# Object-level validation of the fine-tuned U-Net on the 113 annotated patches (results in results/validation/).
#
# Paths default to the Field lab layout on Stanford's Sherlock cluster. On any
# other system, submit from the repository root and override them when submitting, for example:
#   SCRATCH_ROOT=/my/scratch PREPROC_DIR=/my/Preprocessing sbatch slurm/run_object_validation.sbatch
# The annotated patches, imagery and model weights are not public (see README).

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/groups/cfield}"
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"   # submit from the repository root
[ -f "$REPO_DIR/evaluate_object_accuracy.py" ] || REPO_DIR="$REPO_DIR/.."
PREPROC_DIR="${PREPROC_DIR:-/home/groups/cfield/myenv/notebooks/Preprocessing}"
MODEL_DIR="${MODEL_DIR:-$SCRATCH_ROOT/myenv_scratch/notebooks_scratch/saved_models/UNet}"
MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/trees_20260120-1928_AdaDelta_weightmap_tversky_012_256_final.keras}"
TILE_METADATA="${TILE_METADATA:-$SCRATCH_ROOT/tile_metadata.csv}"
CACHED_PREDICTIONS="${CACHED_PREDICTIONS:-$SCRATCH_ROOT/object_validation_results/predictions.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH_ROOT/object_validation_results_tau70}"

ml purge
ml py-tensorflow/2.18.0_py312 viz py-pillow/10.2.0_py312 py-scikit-learn/1.5.1_py312

cd "$SCRATCH_ROOT"
python3 "$REPO_DIR/evaluate_object_accuracy.py" \
    --output_dir "$OUTPUT_DIR" \
    --preproc_dir "$PREPROC_DIR" \
    --tile_metadata_path "$TILE_METADATA" \
    --cached_predictions "$CACHED_PREDICTIONS" \
    --model_dir "$MODEL_DIR" \
    --model_path "$MODEL_PATH" \
    --pred_threshold 0.7 \
    --connectivity 1 \
    --min_crown_area 1
