"""Explicit patient splits and aligned NIfTI inputs for reference training."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
import torchio as tio

from .preprocessing import normalize_image, zscore_image


def read_manifest(path):
    """Read records and reject repeated cases or patients crossing splits.

    Relative file paths are resolved against the manifest, not the working directory.
    ``patient_id`` must identify the person across every included examination.
    """
    path = Path(path).resolve()
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must be a nonempty list")
    patients, cases = {}, set()
    for row in records:
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in cases:
            raise ValueError("Case IDs must be unique nonempty strings")
        cases.add(row["id"])
        patient, split = row.get("patient_id"), row.get("split")
        if not isinstance(patient, str) or not patient or split not in {"train", "val", "test"}:
            raise ValueError("Each record needs patient_id and train/val/test split")
        if patients.setdefault(patient, split) != split:
            raise ValueError(f"Patient crosses splits: {patient}")
        if not isinstance(row.get("channels"), list) or len(row["channels"]) != 3:
            raise ValueError("Supply pre-, early post-, later post-contrast channels in that order")
        row["channels"] = [str((path.parent / name).resolve()) for name in row["channels"]]
        row["label"] = str((path.parent / row["label"]).resolve())
    return records


def prepare_case(row, patch_size=96, spacing_mm=1.0, augment=False):
    """Resample to channel 1, normalize, optionally augment, then center crop.

    This uses an image-center crop, matching the recovered v03 transform.
    The label never determines the crop center.
    """
    paths = [*row["channels"], row["label"]]
    images = [nib.load(path) for path in paths]
    ref = images[0]
    for image in images:
        if len(image.shape) != 3 or image.shape != ref.shape or not np.allclose(image.affine, ref.affine):
            raise ValueError(f"Input grids must already be aligned: {row['id']}")
        if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12:
            raise ValueError("Invalid NIfTI affine")
        if not np.isfinite(image.get_fdata(dtype=np.float32)).all():
            raise ValueError("Nonfinite image or label")
    subject = tio.Subject(
        **{f"ch{i}": tio.ScalarImage(path) for i, path in enumerate(row["channels"])},
        label=tio.LabelMap(row["label"]),
    )
    subject = tio.Resample(spacing_mm)(subject)
    subject = tio.Resample("ch0")(subject)
    volume = np.stack([
        zscore_image(normalize_image(subject[f"ch{i}"][tio.DATA].numpy()[0]))
        for i in range(3)
    ]).astype(np.float32)
    label = (subject["label"][tio.DATA] > .5).float()
    prepared = tio.Subject(image=tio.ScalarImage(tensor=volume), label=tio.LabelMap(tensor=label))
    if augment:
        prepared = tio.Compose([
            tio.RandomFlip(axes=(0, 1, 2)),
            tio.RandomAffine(scales=.1, degrees=10, translation=3, p=.5),
            tio.RandomNoise(std=(0, .05), p=.3),
        ])(prepared)
    prepared = tio.CropOrPad((patch_size,) * 3)(prepared)
    return prepared["image"][tio.DATA].float(), prepared["label"][tio.DATA].float()


class SegmentationDataset(Dataset):
    def __init__(self, records, patch_size=96, spacing_mm=1.0, augment=False):
        if patch_size < 8 or patch_size % 4 or not np.isfinite(spacing_mm) or spacing_mm <= 0:
            raise ValueError("Use patch_size >=8 divisible by four and positive spacing_mm")
        self.records = records
        self.options = dict(patch_size=patch_size, spacing_mm=spacing_mm, augment=augment)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        image, label = prepare_case(row, **self.options)
        return image, label, row["id"]
