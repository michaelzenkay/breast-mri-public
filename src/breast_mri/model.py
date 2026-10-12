"""The segmentation backbone used by the interactive demo."""
from collections.abc import Mapping

import torch
from unet import UNet3D


MODEL_CONFIG = dict(
    in_channels=3,
    out_classes=1,
    num_encoding_blocks=3,
    out_channels_first_layer=32,
    padding=True,
    normalization="batch",
)


def build_model():
    """Three MRI phases -> one tumor logit per voxel; no pCR head."""
    return UNet3D(**MODEL_CONFIG)


def load_model(path, device="cpu"):
    """Load trusted v03 joint-model weights or a reference-trainer checkpoint.

    Historical v03 keys start with ``unet.``; only that backbone is retained.
    Missing or incompatible backbone weights are always an error.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("Expected a checkpoint mapping")
    if "model_state" in state:
        if state.get("model_config") != MODEL_CONFIG:
            raise ValueError("Checkpoint architecture differs from the demo backbone")
        state = state["model_state"]
    elif any(key.startswith("unet.") for key in state):
        state = {key.removeprefix("unet."): value for key, value in state.items()
                 if key.startswith("unet.")}
    model = build_model()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
