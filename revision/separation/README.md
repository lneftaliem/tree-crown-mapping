# Instance separation

Splits the binary U-Net predictions into individual crowns (step 3 of [`REPRODUCING.md`](../../REPRODUCING.md)) with the center-based instance separation of Ankit Kariryaa's [`separate_instances`](https://github.com/ankitkariryaa/separate_instances) (GPL-3.0). The code is included as the git submodule `external/separate_instances`, pinned to commit `d81cac7`.

| File | Purpose |
|---|---|
| `config.sh` | Paths and parameters: 15-pixel maximum filter (`-m 15`), 8-connectivity (`-cn 8`) |
| `00_setup_env.sbatch` | Builds the Python environment (pinned package versions) |
| `01_test_reproduce.sbatch` | Runs the separation on test tiles to check the installation |
| `run_tile.sh <tile> regen` | Separates one prediction tile, checks that the output keeps the input's grid and footprint, and publishes `center_based_relabeled_<tile>` |

The scripts were written for Stanford's Sherlock cluster and expect this folder at `$HOME/split_workflow` and the submodule at `$HOME/software/separate_instances`; set `REPO`, `WORK` and `VENV` in `config.sh` for another system. On a single machine, the equivalent command is:

```bash
cd external/separate_instances
python -m src.center_separation -i <folder of predictions> -p <file stem> -cn 8 -ft .tif -o <output folder> -m 15 -l <log folder>
```
