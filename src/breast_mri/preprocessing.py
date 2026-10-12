"""Intensity normalization, segmentation loss and physical mask measurements."""
import numpy as np
import torch
from torch.nn import functional as F


def normalize_image(arr, min_cutoff=0.001, max_cutoff=0.001):
    """Clip at historical order-statistic cutoffs (default 0.1%) and scale to [0, 1]."""
    flat = np.sort(arr.flatten())
    lo   = flat[int(len(flat) * min_cutoff)]
    hi   = flat[int(len(flat) * max_cutoff) * -1]
    arr  = (arr - lo) / (hi - lo + 1e-8)
    return np.clip(arr, 0.0, 1.0)


def zscore_image(arr):
    """Standardize an array to zero mean and unit variance."""
    return (arr - arr.mean()) / (arr.std() + 1e-8)


def dice_loss(pred, target, smooth=1.0):
    """Soft Dice loss for B1XYZ logits and binary targets, with additive smoothing."""
    pred   = torch.sigmoid(pred)
    inter  = (pred * target).sum(dim=(2, 3, 4))
    denom  = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    return 1.0 - ((2.0 * inter + smooth) / (denom + smooth)).mean()


MEASUREMENT_KEYS = ('d1_mm', 'd2_mm', 'd3_mm', 'volume_mm3', 'sphericity')


def compute_tumor_metrics(mask: np.ndarray,
                          voxel_size_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)
                          ) -> dict[str, float]:
    """Volume, PCA-axis extents and sphericity for a 3D mask thresholded at 0.5.

    Spacing follows the mask's array-axis order. Axes are ordered by decreasing
    PCA variance; their projected extents need not follow that same size order.
    """
    vz, vy, vx = voxel_size_mm
    voxel_vol  = vz * vy * vx

    empty = {k: 0.0 for k in MEASUREMENT_KEYS}

    coords = np.argwhere(mask > 0.5).astype(np.float32)  # (N, 3) voxel indices
    n_vox  = len(coords)
    if n_vox < 4:
        return empty

    # Scale to mm
    coords_mm = coords * np.array([vz, vy, vx], dtype=np.float32)

    centred  = coords_mm - coords_mm.mean(axis=0, keepdims=True)
    cov      = centred.T @ centred / max(n_vox - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)     # ascending order

    # Visit the PCA axes in order of decreasing variance.
    d_mm = []
    for i in [2, 1, 0]:
        axis = eigvecs[:, i]
        proj = centred @ axis
        d_mm.append(float(proj.max() - proj.min()))

    d1_mm, d2_mm, d3_mm = d_mm

    volume_mm3 = float(n_vox * voxel_vol)

    try:
        from skimage.measure import marching_cubes
        verts, faces, _, _ = marching_cubes(mask.astype(np.float32), level=0.5,
                                            spacing=voxel_size_mm)
        v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]
        sa = float(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())
        if not np.isfinite(sa) or sa <= 0:
            raise ValueError('Mask surface area is not positive and finite')
        sphericity = float((np.pi ** (1/3) * (6 * volume_mm3) ** (2/3)) / sa)
    except ImportError as error:
        raise RuntimeError('Sphericity requires scikit-image in the active environment.') from error
    except (ValueError, RuntimeError) as error:
        raise ValueError('Cannot compute sphericity for this mask; no zero feature was substituted.') from error

    return {
        'd1_mm':      d1_mm,
        'd2_mm':      d2_mm,
        'd3_mm':      d3_mm,
        'volume_mm3': volume_mm3,
        'sphericity': sphericity,
    }


def seg_loss_fn(pred, target):
    return dice_loss(pred, target) + F.binary_cross_entropy_with_logits(pred, target)
