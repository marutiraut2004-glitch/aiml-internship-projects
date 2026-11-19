"""Grad-CAM explanations and overlay rendering.

Grad-CAM is implemented directly (forward hook + torch.autograd.grad) instead of using
`monai.visualize.GradCAM`. With this ResNet, MONAI's GradCAM returned a saturated, uninformative map
(median value 0.99, peak stuck at pixel (0, 0), 0 of 60 peaks within 12 px of a known lesion) while this
implementation of the same algorithm, on the same layer of the same model, put the peak within 12 px of the
lesion 95% of the time. See tests/test_imaging.py::test_gradcam_localises_inserted_lesion.
"""
from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .model import TARGET_LAYER


def heatmap(model: torch.nn.Module, x: torch.Tensor, class_idx: int = 1, layer: str = TARGET_LAYER) -> np.ndarray:
    """Grad-CAM. (1,1,H,W) -> (H,W) in [0,1] for `class_idx` (default: the abnormal class).

    cam = ReLU( sum_k  mean_xy(d logit / d A_k)  *  A_k ),  bilinearly upsampled to the input size."""
    model.eval()
    store = {}
    handle = dict(model.named_modules())[layer].register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    try:
        with torch.enable_grad():
            logits = model(x.clone().requires_grad_(False))
            act = store["a"]
            grad = torch.autograd.grad(logits[0, class_idx], act)[0]
    finally:
        handle.remove()
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(1, keepdim=True)).detach()
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].numpy()
    return (cam / cam.max()).astype("float32") if cam.max() > 0 else np.zeros_like(cam, dtype="float32")


def _jet(v: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], -1)


def overlay(img: np.ndarray, heat: np.ndarray, alpha: float = 0.45, scale: int = 3) -> bytes:
    """PNG bytes of the image with the heatmap blended on top (heat < 0.2 stays transparent)."""
    base = np.repeat(img[..., None], 3, -1)
    a = alpha * np.clip((heat - 0.2) / 0.8, 0, 1)[..., None]
    out = (1 - a) * base + a * _jet(heat)
    im = Image.fromarray((out * 255).astype("uint8")).resize((img.shape[1] * scale, img.shape[0] * scale), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def grey_png(img: np.ndarray, scale: int = 3) -> bytes:
    im = Image.fromarray((img * 255).astype("uint8")).resize((img.shape[1] * scale, img.shape[0] * scale), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def pointing_hit(heat: np.ndarray, lesion, margin: float = 4.0) -> bool:
    """Pointing game: does the heatmap's peak fall inside the known lesion (plus a small margin)?"""
    cx, cy, r = lesion
    y, x = np.unravel_index(np.argmax(heat), heat.shape)
    return bool((x - cx) ** 2 + (y - cy) ** 2 <= (r + margin) ** 2)
