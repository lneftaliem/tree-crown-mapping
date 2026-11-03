#!/bin/bash
# Object-level validation on the 113 annotated patches (Figures S1-S3), original protocol and final v1 pipeline.
# Writes only $GROUP_SCRATCH/leonan_split_2026/validation_v1/<run>/ ; reads the annotation frames (read-only) and the
# cached original predictions. Runs (all with code/analysis/repo_adapted/evaluate_object_accuracy.py):
#   A   NDVI+PAN inputs, threshold 0.3, connected components  -> reproduces the published SI numbers (cached predictions)
#   A7  NDVI+PAN inputs, threshold 0.7, connected components  -> the threshold stated in the SI captions
#   B   PAN+PAN inputs (citywide inference), 0.7, connected components (semantic baseline of the final pipeline)
#   C0  PAN+PAN, 0.7, instance-separated (separate_instances -cn 8 -m 15, as citywide), no area filter
#   C   PAN+PAN, 0.7, instance-separated, crowns >= 3 m2  -> the final v1 pipeline
#SBATCH --job-name=val_v1
#SBATCH --partition=serc
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
V=$GROUP_SCRATCH/leonan_split_2026/validation_v1
EVAL=$HOME/unet_workflow/validation/evaluate_object_accuracy.py
META=$HOME/unet_workflow/validation/tile_metadata.csv
PRE=/home/groups/cfield/myenv/notebooks/Preprocessing
MD=$GROUP_SCRATCH/myenv_scratch/notebooks_scratch/saved_models/UNet
MODEL=$MD/trees_20260120-1928_AdaDelta_weightmap_tversky_012_256_final.keras
CACHE=$GROUP_SCRATCH/object_validation_results/predictions.npy
mkdir -p $V
ml purge; ml py-tensorflow/2.18.0_py312 viz py-pillow/10.2.0_py312 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=-1
python3 - <<'EOF' || { echo "PREFLIGHT FAILED on $(hostname) job ${SLURM_JOB_ID}"; exit 3; }
import socket, hashlib, ssl, certifi
p = certifi.where()
print('preflight', socket.gethostname(), p, hashlib.sha1(open(p, 'rb').read()).hexdigest(), ssl.OPENSSL_VERSION, flush=True)
ssl.create_default_context().load_verify_locations(p)
import tensorflow; print('preflight tensorflow', tensorflow.__version__, 'ok', flush=True)
EOF
COMMON="--preproc_dir $PRE --tile_metadata_path $META --model_dir $MD --model_path $MODEL --connectivity 1 --min_crown_area 1"
cd $V
python3 $EVAL $COMMON --output_dir $V/A  --cached_predictions $CACHE --pred_threshold 0.3 --input_channels ndvi_pan > A.log 2>&1;  echo "A exit $?"
python3 $EVAL $COMMON --output_dir $V/A7 --cached_predictions $CACHE --pred_threshold 0.7 --input_channels ndvi_pan > A7.log 2>&1; echo "A7 exit $?"
python3 $EVAL $COMMON --output_dir $V/B  --pred_threshold 0.7 --input_channels pan_pan --export_binary_dir $V/B/binary > B.log 2>&1;  echo "B exit $?"

# instance separation of the PAN+PAN binarized patches, Christian's parameters (as citywide)
ml purge; ml python/3.12.1; source $HOME/envs/separate_instances/bin/activate
source $HOME/split_workflow/config.sh
mkdir -p $V/sep_out $V/sep_log
( cd $REPO && python -m src.center_separation -i $V/B/binary -p pred_patch -cn $CONNECTIVITY -ft $FILE_TYPE -o $V/sep_out -m $MAX_FILTER -l $V/sep_log ) > sep.log 2>&1; echo "sep exit $?"
python3 - <<'EOF' || exit 5
import os, glob, numpy as np, rasterio
V = os.environ['GROUP_SCRATCH'] + '/leonan_split_2026/validation_v1'
n = len(glob.glob(f'{V}/B/binary/pred_patch_*.tif'))
labs = []
for i in range(n):
    with rasterio.open(f'{V}/sep_out/center_based_relabeled_pred_patch_{i:03d}.tif') as s:
        labs.append(s.read(1).astype(np.int32))
np.save(f'{V}/instance_labels.npy', np.stack(labs))
print('instance labels', np.stack(labs).shape, 'instances', sum(len(np.unique(l)) - 1 for l in labs))
EOF

ml purge; ml py-tensorflow/2.18.0_py312 viz py-pillow/10.2.0_py312 py-scikit-learn/1.5.1_py312
python3 $EVAL $COMMON --output_dir $V/C0 --cached_predictions $V/B/predictions.npy --pred_threshold 0.7 --input_channels pan_pan \
  --instance_labels_npy $V/instance_labels.npy > C0.log 2>&1; echo "C0 exit $?"
python3 $EVAL $COMMON --output_dir $V/C  --cached_predictions $V/B/predictions.npy --pred_threshold 0.7 --input_channels pan_pan \
  --instance_labels_npy $V/instance_labels.npy --min_crown_m2 3.0 > C.log 2>&1; echo "C exit $?"
