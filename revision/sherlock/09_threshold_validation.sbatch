#!/bin/bash
# Threshold selection for the final NDVI+PAN configuration (113 annotated patches; protocol of evaluate_object_accuracy.py).
# Writes only $GROUP_SCRATCH/leonan_split_2026/threshold_v1/.
#   1. fresh NDVI+PAN predictions of the 113 patches (model trees_20260120-1928_..._final.keras), compared with the cached
#      original predictions (object_validation_results/predictions.npy) for reproducibility;
#   2. for tau in 0.3 / 0.5 / 0.7: binarize, separate instances with Christian's separate_instances (-cn 8 -m 15, as
#      citywide), and evaluate three object definitions with everything else identical:
#        cc_<tau>      4-connected components (semantic-segmentation objects, the original protocol)
#        inst_<tau>    instance-separated crowns
#        final_<tau>   instance-separated crowns >= 3 m2 (the final dataset's definition)
#SBATCH --job-name=thr_v1
#SBATCH --partition=serc
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
V=$GROUP_SCRATCH/leonan_split_2026/threshold_v1
EVAL=$HOME/unet_workflow/validation/evaluate_object_accuracy.py
META=$HOME/unet_workflow/validation/tile_metadata.csv
PRE=/home/groups/cfield/myenv/notebooks/Preprocessing
MD=$GROUP_SCRATCH/myenv_scratch/notebooks_scratch/saved_models/UNet
MODEL=$MD/trees_20260120-1928_AdaDelta_weightmap_tversky_012_256_final.keras
mkdir -p $V; cd $V
tfenv() { ml purge; ml py-tensorflow/2.18.0_py312 viz py-pillow/10.2.0_py312 py-scikit-learn/1.5.1_py312; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=-1; }
tfenv
python3 - <<'EOF' || { echo "PREFLIGHT FAILED on $(hostname) job ${SLURM_JOB_ID}"; exit 3; }
import socket, hashlib, ssl, certifi
p = certifi.where()
print('preflight', socket.gethostname(), p, hashlib.sha1(open(p, 'rb').read()).hexdigest(), ssl.OPENSSL_VERSION, flush=True)
ssl.create_default_context().load_verify_locations(p)
import tensorflow; print('preflight tensorflow', tensorflow.__version__, 'ok', flush=True)
EOF
COMMON="--preproc_dir $PRE --tile_metadata_path $META --model_dir $MD --model_path $MODEL --connectivity 1 --min_crown_area 1 --input_channels ndvi_pan"
# 1. fresh predictions (+ CC evaluation and binary export at 0.7)
python3 $EVAL $COMMON --output_dir $V/cc_0.7 --pred_threshold 0.7 --export_binary_dir $V/bin_0.7 > cc_0.7.log 2>&1; echo "cc_0.7 exit $?"
PRED=$V/cc_0.7/predictions.npy
python3 - <<EOF
import numpy as np
a, b = np.load('$PRED'), np.load('$GROUP_SCRATCH/object_validation_results/predictions.npy')
print('fresh', a.shape, 'cached', b.shape)
if b.shape[0] == 114: b = np.delete(b, 80, axis=0)  # the evaluator's documented 114 -> 113 mapping (uncurated patch 344)
print('max |fresh - cached| =', float(np.abs(a - np.squeeze(b)).max()))
EOF
for T in 0.3 0.5; do
  python3 $EVAL $COMMON --output_dir $V/cc_$T --cached_predictions $PRED --pred_threshold $T --export_binary_dir $V/bin_$T > cc_$T.log 2>&1; echo "cc_$T exit $?"
done
# 2. instance separation of each threshold's binarized patches
ml purge; ml python/3.12.1; source $HOME/envs/separate_instances/bin/activate; source $HOME/split_workflow/config.sh
for T in 0.3 0.5 0.7; do
  mkdir -p $V/sep_$T $V/seplog_$T
  ( cd $REPO && python -m src.center_separation -i $V/bin_$T -p pred_patch -cn $CONNECTIVITY -ft $FILE_TYPE -o $V/sep_$T -m $MAX_FILTER -l $V/seplog_$T ) > sep_$T.log 2>&1
  python3 - $T <<'EOF'
import os, sys, glob, numpy as np, rasterio
T = sys.argv[1]; V = os.environ['GROUP_SCRATCH'] + '/leonan_split_2026/threshold_v1'
n = len(glob.glob(f'{V}/bin_{T}/pred_patch_*.tif')); labs = []
for i in range(n):
    with rasterio.open(f'{V}/sep_{T}/center_based_relabeled_pred_patch_{i:03d}.tif') as s:
        labs.append(s.read(1).astype(np.int32))
np.save(f'{V}/instances_{T}.npy', np.stack(labs)); print('tau', T, 'patches', n, 'instances', sum(len(np.unique(l)) - 1 for l in labs))
EOF
done
tfenv
for T in 0.3 0.5 0.7; do
  python3 $EVAL $COMMON --output_dir $V/inst_$T  --cached_predictions $PRED --pred_threshold $T --instance_labels_npy $V/instances_$T.npy > inst_$T.log 2>&1;  echo "inst_$T exit $?"
  python3 $EVAL $COMMON --output_dir $V/final_$T --cached_predictions $PRED --pred_threshold $T --instance_labels_npy $V/instances_$T.npy --min_crown_m2 3.0 > final_$T.log 2>&1; echo "final_$T exit $?"
done
