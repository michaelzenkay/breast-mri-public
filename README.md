# Breast MRI tumor segmentation

3D U-Net methods for three-phase DCE breast MRI: training, evaluation,
whole-volume inference and seed-centered segmentation. Developed for I-SPY1
MRI; images, labels, trained weights and the viewer are not included.

## Method

Inputs are three aligned 3D NIfTI images, ordered pre-contrast, early
post-contrast and later post-contrast. Each phase is clipped/scaled independently
and z-scored. DICOM conversion and image registration are outside this package.

The [model](src/breast_mri/model.py) uses three input channels, three encoding
blocks, first-layer width 32, batch normalization and one output logit per voxel.
Tumor probabilities are thresholded at >0.5.

## Install

Requires Python 3.11+. Install a suitable PyTorch build first if you need CUDA.

```bash
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install .
```

## Inference

```python
import nibabel as nib
from breast_mri.model import load_model
from breast_mri.inference import load_volume, predict_volume_single_pass

device = "cuda"  # Or "cpu".
model = load_model("checkpoint.pth", device)
image, grid = load_volume(["pre.nii.gz", "early.nii.gz", "late.nii.gz"], spacing_mm=1.0)
probability = predict_volume_single_pass(image, model, device)
nib.save(nib.Nifti1Image((probability > 0.5).astype("uint8"), grid.affine), "mask.nii.gz")
```

[Inference](src/breast_mri/inference.py) also provides:

- `predict_seed`: load the native grid with `load_volume(paths)`, convert a
  physical point using `world_seed(grid, xyz_mm)`, then call
  `predict_seed(image, seed, model, device)`. Returns a mask and inference time
  in milliseconds. Only the 96 x 96 x 96 voxel crop around the seed is evaluated.
- `predict_volume`: lower-memory whole-volume inference using overlapping
  96 x 96 x 96 voxel patches and Hann-weighted logit blending.

`compute_tumor_metrics(mask, voxel_size_mm)` in
[preprocessing.py](src/breast_mri/preprocessing.py) returns volume in mm^3,
extents along PCA axes in mm, and sphericity. Spacing follows the mask's array axes.
Masks with fewer than four foreground voxels return zero metrics.

## Train and evaluate

Edit the [case manifest](configs/cases.example.json) with your images, binary
labels, patient IDs and train/validation splits, then set the
[training configuration](configs/reference.json). Patients cannot cross splits.

```bash
breast-mri-train --config configs/reference.json --output runs/reference --device cuda
breast-mri-evaluate --manifest configs/cases.example.json --checkpoint runs/reference/best.pt --device cuda
```

Default preprocessing resamples to 1 mm and crops/pads to 96 x 96 x 96 voxels
at the image center. Training optionally applies flips, affine transforms and noise;
validation has no augmentation. The segmentation-only trainer uses BCE plus
soft Dice, Adam and cosine learning-rate decay.

The best checkpoint is selected by mean per-case Dice. A new output directory
receives `best.pt`, `last.pt`, `metrics.csv` and `run_config.json`. Evaluation
reports per-case Dice and its mean; two empty masks receive 1. If you change
training spacing or crop size, pass matching `--spacing-mm` and `--patch-size`
options to evaluation.

## Tests

```bash
python -B -m unittest discover -s src/tests -v
```

Three synthetic CPU tests cover inference, physical volume and training/reload.
They use the installed package; reinstall after source changes. These are
software checks, not model-accuracy measurements.

## License and attribution

Source: [Apache-2.0](LICENSE). U-Net layers: Fernando Perez-Garcia's
[unet](https://github.com/fepegar/unet) (MIT). Resampling and augmentation:
[TorchIO](https://github.com/TorchIO-project/torchio) (Apache-2.0).
I-SPY1 data originate from The Cancer Imaging Archive; data and weights retain
separate terms and are not covered by this source license.
