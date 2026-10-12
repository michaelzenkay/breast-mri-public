"""Single-device segmentation reference trainer, distinct from joint v03 training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SegmentationDataset, read_manifest
from .evaluation import evaluate
from .model import MODEL_CONFIG, build_model, load_model
from .preprocessing import seg_loss_fn


def train(config_path, output, device="cpu"):
    config_path, output = Path(config_path).resolve(), Path(output)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = (config_path.parent / config["manifest"]).resolve()
    records = read_manifest(manifest)
    epochs, batch_size = config["epochs"], config["batch_size"]
    if epochs < 1 or batch_size < 1 or not np.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("Positive epochs, batch_size and learning_rate required")
    splits = {split: [r for r in records if r["split"] == split] for split in ["train", "val"]}
    if not all(splits.values()):
        raise ValueError("Both train and val patients are required")
    seed = config.get("seed", 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    options = dict(patch_size=config.get("patch_size", 96), spacing_mm=config.get("spacing_mm", 1.))
    datasets = {split: SegmentationDataset(rows, **options, augment=split == "train" and config.get("augment", True))
                for split, rows in splits.items()}
    loaders = {split: DataLoader(dataset, batch_size=batch_size, shuffle=split == "train", num_workers=0)
               for split, dataset in datasets.items()}
    device = torch.device(device)
    initial = config.get("initial_checkpoint")
    model = load_model(config_path.parent / initial, device) if initial else build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=config.get("min_lr", 1e-6))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    # Never overwrite a previous experiment directory.
    output.mkdir(parents=True, exist_ok=False)
    record = dict(config, model_config=MODEL_CONFIG, scope="segmentation-only reference training",
                  manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  selection_metric="mean_case_dice", preprocessing=options,
                  patient_counts={s: len({r['patient_id'] for r in rows}) for s, rows in splits.items()},
                  torch_version=str(torch.__version__), device=str(device))
    if initial:
        record["initial_checkpoint_sha256"] = hashlib.sha256((config_path.parent / initial).read_bytes()).hexdigest()
    (output / "run_config.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    best = -1.
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "mean_case_dice"])
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for images, labels, _ in loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    loss = seg_loss_fn(model(images.to(device)), labels.to(device))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
                losses.append(float(loss.detach()))
            result = evaluate(model, loaders["val"], device)
            row = dict(epoch=epoch, train_loss=sum(losses) / len(losses),
                       mean_case_dice=result["mean_case_dice"])
            writer.writerow(row); stream.flush()
            scheduler.step()
            checkpoint = dict(model_config=MODEL_CONFIG, model_state=model.state_dict(), epoch=epoch,
                              selection_metric="mean_case_dice", validation=result)
            torch.save(checkpoint, output / "last.pt")
            if result["mean_case_dice"] > best:
                best = result["mean_case_dice"]
                torch.save(checkpoint, output / "best.pt")
            print(json.dumps(row))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(args.config, args.output, args.device)


if __name__ == "__main__":
    main()
