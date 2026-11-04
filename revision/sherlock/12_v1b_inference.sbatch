#!/bin/bash
# v1b citywide inference (v1 with input channels in training order: band 1 PAN, band 2 NDVI; see regen_v1b.py) (NDVI + PAN, threshold $THRESH) + instance separation, one task per scene
# (v1_chunks/v1_NNN.txt: every in-city window of the 23 scenes, one representative file name per window).
# Writes only $GROUP_SCRATCH/leonan_split_2026/v1b_pan_ndvi/. Submit with: sbatch --export=ALL,THRESH=<tau> 12_v1b_inference.sbatch
#SBATCH --job-name=v1b_infer
#SBATCH --partition=serc
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-23%12
#SBATCH --output=logs/%x_%A_%a.out
set -uo pipefail
: "${THRESH:?set THRESH}"
NB=$GROUP_SCRATCH/myenv_scratch/notebooks_scratch
WORK=$GROUP_SCRATCH/leonan_split_2026
V1=$WORK/v1b_pan_ndvi
export NB_DIR=$NB RAW_DIR=$NB/raw_images THRESH
export MODEL=$NB/saved_models/UNet/trees_20260120-1928_AdaDelta_weightmap_tversky_012_256_final.keras
ID=$(printf %03d $SLURM_ARRAY_TASK_ID)
ml purge; ml math py-tensorflow/2.18.0_py312
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=-1
export PYTHONPATH=$HOME/unet_workflow:${PYTHONPATH:-}
python3 - <<'EOF' || { echo "PREFLIGHT FAILED on $(hostname) job ${SLURM_JOB_ID}"; exit 3; }
import socket, hashlib, ssl, certifi
p = certifi.where()
print('preflight', socket.gethostname(), p, hashlib.sha1(open(p, 'rb').read()).hexdigest(), ssl.OPENSSL_VERSION, flush=True)
ssl.create_default_context().load_verify_locations(p)
import tensorflow; print('preflight tensorflow', tensorflow.__version__, 'ok', flush=True)
EOF
python3 $HOME/unet_workflow/regen_v1b.py $HOME/unet_workflow/v1_chunks/v1_$ID.txt $V1 $L_SCRATCH_JOB || exit 4

# instance separation (separate_instances by Christian Igel and Ankit Kariryaa, -cn 8 -m 15, via the verified run_tile.sh wrapper)
ml purge; ml python/3.12.1; source $HOME/envs/separate_instances/bin/activate
export REGEN=$V1/predictions OUT=$V1/split CENTERS=$V1/centers
mkdir -p $OUT $CENTERS
cut -d' ' -f1 $HOME/unet_workflow/v1_chunks/v1_$ID.txt | while read f; do [ -f $REGEN/$f ] && echo $f; done \
  | xargs -P $SLURM_CPUS_PER_TASK -I{} bash $HOME/split_workflow/run_tile.sh {} regen
