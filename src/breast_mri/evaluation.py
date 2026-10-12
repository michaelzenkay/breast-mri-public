"""Per-case Dice and its mean across the evaluation cohort."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import SegmentationDataset, read_manifest
from .model import load_model


@torch.inference_mode()
def evaluate(model, loader, device="cpu"):
    model.eval()
    cases = []
    for images, targets, case_ids in loader:
        images, targets = images.to(device), targets.to(device)
        with torch.autocast(device_type=torch.device(device).type, enabled=torch.device(device).type == "cuda"):
            logits = model(images)
        if logits.shape != targets.shape or not torch.isfinite(logits).all():
            raise ValueError("Invalid model output")
        predicted = (torch.sigmoid(logits) > .5).float()
        intersection = (predicted * targets).flatten(1).sum(1)
        denominator = predicted.flatten(1).sum(1) + targets.flatten(1).sum(1)
        for case_id, inter, denom in zip(case_ids, intersection.tolist(), denominator.tolist()):
            cases.append({"id": case_id, "dice": 2 * inter / denom if denom else 1.0})
    if not cases:
        raise ValueError("Evaluation split is empty")
    return {
        "case_count": len(cases),
        "mean_case_dice": sum(row["dice"] for row in cases) / len(cases),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--spacing-mm", type=float, default=1.)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    records = [row for row in read_manifest(args.manifest) if row["split"] == args.split]
    dataset = SegmentationDataset(records, args.patch_size, args.spacing_mm)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    result = evaluate(load_model(args.checkpoint, args.device), loader, args.device)
    result.update(split=args.split, batch_size=args.batch_size,
                  patch_size=args.patch_size, spacing_mm=args.spacing_mm)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
