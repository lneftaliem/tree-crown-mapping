# U-Net fine-tuning

Training-data preparation and fine-tuning of the U-Net used in the revised manuscript (step 1 of [`REPRODUCING.md`](../../REPRODUCING.md)).

| File | Purpose |
|---|---|
| `1-Preprocessing.ipynb`, `config/Preprocessing.py` | Cuts the 113 annotated training areas from the imagery and writes the image bands, crown annotations and boundary weight maps |
| `2-UNetTraining.ipynb`, `2-UNetTraining.py`, `config/UNetTraining.py` | Fine-tunes the U-Net from the Sahel/Sudan weights of Brandt et al. (2020) (set `PRETRAINED_WEIGHTS`) |
| `config/RasterAnalysis.py` | Settings for tile-based inference with `3-RasterAnalysis.py` (original submission) |
| `core/` | U-Net architecture, losses (weighted Tversky), optimizers, data generators and augmentation |

Notebook outputs have been removed because they contain licensed imagery. The fine-tuned model is attached to the [v1.0 release](https://github.com/lneftaliem/tree-crown-mapping/releases/tag/v1.0).

`core/` and `config/` are adapted from [ankitkariryaa/An-unexpectedly-large-count-of-trees-in-the-western-Sahara-and-Sahel](https://github.com/ankitkariryaa/An-unexpectedly-large-count-of-trees-in-the-western-Sahara-and-Sahel) (MIT licence; Brandt et al., 2020).
