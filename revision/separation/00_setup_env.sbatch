#!/bin/bash
# Create the Python environment for separate_instances (pinned to the repo's uv.lock versions).
#SBATCH --job-name=si_setup
#SBATCH --partition=serc
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=%x_%j.out
set -euo pipefail
source $HOME/split_workflow/config.sh
ml purge
ml python/3.12.1
export PIP_CACHE_DIR=$SCRATCH/.cache/pip XDG_CACHE_HOME=$SCRATCH/.cache
cd $REPO && git checkout -q $REPO_COMMIT && git log --oneline -1
rm -rf $VENV
python3 -m venv $VENV
source $VENV/bin/activate
pip install --quiet --upgrade pip
# uv.lock at d81cac7 pins numpy 2.5.4 / scipy 1.18.1 / numba 0.68.1 ..., which have no wheels for Sherlock's
# CentOS 7 (glibc 2.17). Use the newest manylinux2014 (glibc 2.17) wheels instead; never build from source.
# Equivalence with the rasters supplied by Christian Igel is verified pixel-for-pixel by 01_test_reproduce.sbatch.
pip install --quiet --only-binary=:all: numpy==2.2.6 scipy==1.15.3 numba==0.61.2 llvmlite==0.44.0 \
    rasterio==1.4.3 geopandas==1.0.1 shapely==2.0.7 pyogrio==0.10.0 pandas==2.2.3 pyproj==3.7.1 \
    psutil==7.0.0 joblib==1.5.1 tqdm==4.67.1
pip freeze > $VENV/pip_freeze.txt
python -c "import numpy, scipy, numba, rasterio, geopandas; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'numba', numba.__version__, 'rasterio', rasterio.__version__)"
cd $REPO && python -m src.center_separation -h | head -5
du -sh $VENV
