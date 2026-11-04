#!/bin/bash
# run_tile.sh <tile_name> <source: zip|cutouts|regen>   -- one tile, the exact CLI and parameters of separate_instances (Christian Igel and Ankit Kariryaa).
# Stages the input on node-local disk, runs the repo CLI, checks the result, and publishes it
# atomically (copy to .partial, then rename), so an interrupted job never leaves a truncated output.
set -uo pipefail
source $HOME/split_workflow/config.sh
NAME=$1; SRC=$2; STEM=${NAME%.tif}
FINAL=$OUT/center_based_relabeled_$NAME
[ -s "$FINAL" ] && { echo "SKIP $NAME (exists)"; exit 0; }
TMP=${L_SCRATCH_JOB:-$TMPDIR}/si_$STEM; rm -rf $TMP; mkdir -p $TMP/in $TMP/out $TMP/log
if [ "$SRC" = zip ]; then unzip -q -j "$ARCHIVE" "$NAME" -d $TMP/in || exit 11
elif [ "$SRC" = regen ]; then cp "$REGEN/$NAME" $TMP/in/ || exit 11
else cp "$CUTOUTS/$NAME" $TMP/in/ || exit 11; fi
cd $REPO
python -m src.center_separation -i $TMP/in -p "$STEM" -cn $CONNECTIVITY -ft $FILE_TYPE -o $TMP/out -m $MAX_FILTER -l $TMP/log || exit 12
python - "$TMP/in/$NAME" "$TMP/out/center_based_relabeled_$NAME" <<'PY' || exit 13
import sys, rasterio, numpy as np
from scipy import ndimage
with rasterio.open(sys.argv[1]) as a, rasterio.open(sys.argv[2]) as b:
    assert (a.width, a.height) == (b.width, b.height) and a.transform == b.transform and a.crs == b.crs, 'grid mismatch'
    i, o = a.read(1, masked=True).filled(0) > 0, b.read(1)
    assert np.array_equal(o > 0, ndimage.binary_fill_holes(i)), 'footprint != hole-filled input'
    print('CHECK ok', sys.argv[2].split('/')[-1], 'labels', int(o.max()), 'px', int((o > 0).sum()))
PY
cp $TMP/out/center_based_relabeled_$NAME $FINAL.partial && mv $FINAL.partial $FINAL || exit 14
cp $TMP/out/centers_$NAME $CENTERS/ 2>/dev/null
rm -rf $TMP
echo "DONE $NAME"
