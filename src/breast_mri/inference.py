"""Seed-crop and full-volume inference, independent of any web framework."""
import time

import nibabel as nib
import numpy as np
import torch
from .preprocessing import normalize_image, zscore_image


def load_volume(paths, spacing_mm=None):
    """Load three aligned phases, optionally resample, then normalize each phase.

    Returns a float32 CXYZ array and a NIfTI image describing the output grid.
    Set spacing_mm=1.0 for the saved full-volume demo; omit it for native seed inference.
    """
    if len(paths) != 3:
        raise ValueError('Exactly three channels required')
    channels = [nib.load(str(path)) for path in paths]
    ref = channels[0]
    if len(ref.shape) != 3 or any(c.shape != ref.shape or not np.allclose(c.affine, ref.affine) for c in channels):
        raise ValueError('Channel grid mismatch')
    if not np.isfinite(ref.affine).all() or abs(np.linalg.det(ref.affine[:3, :3])) < 1e-12:
        raise ValueError('Invalid image affine')
    arrays = [c.get_fdata(dtype=np.float32) for c in channels]
    if not all(np.isfinite(arr).all() for arr in arrays):
        raise ValueError('Nonfinite image')
    affine = ref.affine
    if spacing_mm is not None:
        import torchio as tio
        if not np.isfinite(spacing_mm) or spacing_mm <= 0:
            raise ValueError('spacing_mm must be positive')
        subject = tio.Subject(**{f'ch{i}': tio.ScalarImage(path) for i, path in enumerate(paths)})
        subject = tio.Resample('ch0')(tio.Resample(spacing_mm)(subject))
        arrays = [subject[f'ch{i}'].data.numpy()[0] for i in range(3)]
        affine = subject.ch0.affine
    image = np.stack([zscore_image(normalize_image(arr)) for arr in arrays]).astype(np.float32)
    return image, nib.Nifti1Image(arrays[0], affine)


def crop_bounds(shape, seed, patch_size=96):
    origin = np.asarray(seed) - patch_size // 2
    return origin, np.maximum(origin, 0), np.minimum(origin + patch_size, shape)


def world_seed(ref, seed_mm):
    native = nib.affines.apply_affine(np.linalg.inv(ref.affine), seed_mm)
    # Reject outside continuous coordinates before rounding, allowing numerical noise.
    upper = np.asarray(ref.shape) - 1
    if not np.isfinite(native).all() or np.any(native < -1e-5) or np.any(native > upper + 1e-5):
        raise ValueError('Seed outside image voxel centers')
    return tuple(int(v) for v in np.floor(np.clip(native, 0, upper) + 0.5))


def predict_seed(image, seed, model, device, patch_size=96):
    if image.ndim != 4 or image.shape[0] != 3 or not np.isfinite(image).all():
        raise ValueError('Expected a finite three-channel CXYZ image')
    seed = np.asarray(seed)
    if seed.shape != (3,) or not np.isfinite(seed).all() or np.any(seed != np.floor(seed)):
        raise ValueError('Seed must contain three integer voxel coordinates')
    if np.any(seed < 0) or np.any(seed >= np.asarray(image.shape[1:])):
        raise ValueError('Seed outside image')
    if patch_size < 8 or patch_size % 4:
        raise ValueError('Use patch_size >=8 divisible by four')
    # Keep the seed at the crop center even at image edges; pad outside with zero.
    shape = np.asarray(image.shape[1:])
    origin, low, high = crop_bounds(shape, seed, patch_size)
    src = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    dst = tuple(slice(int(a), int(b)) for a, b in zip(low - origin, high - origin))
    crop = np.zeros((3, patch_size, patch_size, patch_size), dtype=np.float32)
    crop[(slice(None),) + dst] = image[(slice(None),) + src]
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(torch.from_numpy(crop[None]).to(device))
        predicted = (logits[0, 0] > 0).cpu().numpy()
    inference_ms = (time.perf_counter() - start) * 1000
    mask = np.zeros(tuple(shape), dtype=np.uint8)
    mask[src] = predicted[dst]
    return mask, inference_ms


def predict_volume(image, model, device, patch_size=96, overlap=48):
    """Predict every voxel of a normalized CXYZ volume with Hann-blended logits.

    The caller owns spatial resampling and normalization. Zero-pad normalized
    intensities at the image boundary; return probabilities on the input grid.
    No seed, training label, or component filtering is used.
    """
    import torchio as tio

    if image.ndim != 4 or image.shape[0] != 3 or not np.isfinite(image).all():
        raise ValueError('Expected a finite three-channel CXYZ image')
    if patch_size < 8 or patch_size % 4 or overlap <= 0 or overlap >= patch_size or overlap % 2:
        raise ValueError('Use a patch size divisible by four and positive even overlap smaller than the patch')
    native_shape = image.shape[1:]
    gaps = [max(0, patch_size - n) for n in native_shape]
    padding = [(n // 2, n - n // 2) for n in gaps]
    padded = np.pad(image.astype(np.float32), [(0, 0), *padding])
    subject = tio.Subject(image=tio.ScalarImage(tensor=padded))
    sampler = tio.GridSampler(subject, patch_size, overlap, padding_mode=0)
    aggregator = tio.GridAggregator(sampler, overlap_mode='hann')
    loader = torch.utils.data.DataLoader(sampler, batch_size=1)
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch['image'][tio.DATA].to(device))
            aggregator.add_batch(logits.cpu(), batch[tio.LOCATION])
    logits = aggregator.get_output_tensor()[0]
    region = tuple(slice(before, before + n) for (before, _), n in zip(padding, native_shape))
    probability = torch.sigmoid(logits[region]).numpy()
    if probability.shape != native_shape or not np.isfinite(probability).all():
        raise RuntimeError('Invalid full-volume output')
    return probability


def predict_volume_single_pass(image, model, device):
    """Evaluate the complete image, with no internal patch boundaries.

    Pad to the two-pooling-stage U-Net's multiple of four. CUDA autocast reduces
    activation memory; output probabilities are converted to float32.
    """
    if image.ndim != 4 or image.shape[0] != 3 or not np.isfinite(image).all():
        raise ValueError('Expected a finite three-channel CXYZ image')
    shape = image.shape[1:]
    padding = [(0, (-n) % 4) for n in shape]
    padded = np.pad(image.astype(np.float32), [(0, 0), *padding])
    device = torch.device(device)
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
        logits = model(torch.from_numpy(padded[None]).to(device))
    probability = torch.sigmoid(logits[0, 0].float()).cpu().numpy()
    probability = probability[tuple(slice(0, n) for n in shape)]
    if probability.shape != shape or not np.isfinite(probability).all():
        raise RuntimeError('Invalid full-volume output')
    return probability
