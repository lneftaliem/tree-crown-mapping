# Shared configuration for the center-based instance-separation workflow on Sherlock.
# Sourced by every job script. Paths use Sherlock environment variables (see /etc/agents/storage.md).
WORK=$GROUP_SCRATCH/leonan_split_2026            # inputs, outputs, logs (purged after 90 days: copy results out)
REPO=$HOME/software/separate_instances           # github.com/ankitkariryaa/separate_instances (written by Christian Igel and Ankit Kariryaa)
REPO_COMMIT=d81cac7                              # commit verified pixel-for-pixel against the rasters supplied by Christian Igel
VENV=$HOME/envs/separate_instances               # environments never go in $SCRATCH
ARCHIVE=$GROUP_SCRATCH/myenv_scratch/notebooks_scratch/used_predictions.zip
CUTOUTS=$GROUP_SCRATCH/myenv_scratch/notebooks_scratch/cutouts
OUT=${OUT:-$WORK/outputs/center_based_relabeled}   # final instance rasters (overridable for tests)
CENTERS=${CENTERS:-$WORK/outputs/centers}         # by-product of the repo CLI, kept for provenance
# Parameters set by Christian Igel -- do not change without discussion
MAX_FILTER=15
CONNECTIVITY=8
FILE_TYPE=.tif
REGEN=${REGEN:-$WORK/regen/predictions}   # regenerated U-Net predictions (unet_workflow/02_regen_array.sbatch)
