from __future__ import annotations

import numpy as np
import torch
from monai.networks.nets import ResNet

CLASSES = ["normal", "nodule"]
TARGET_LAYER = "layer4"


def build_model() -> torch.nn.Module:
    """Small MONAI ResNet that keeps spatial resolution (stride-1 stem, no max-pool).

    The final feature map is 16x16 and each cell sees a ~49 px window, so Grad-CAM on `layer4` can say
    *where* a 5-9 px lesion is. A DenseNet with the standard /32 downsampling scored the same AUC but its
    4x4 feature map could not localise anything (0 of 100 Grad-CAM peaks landed on the lesion)."""
    return ResNet(block="basic", layers=[1, 1, 1, 1], block_inplanes=[16, 32, 64, 128], spatial_dims=2,
                  n_input_channels=1, num_classes=2, conv1_t_size=3, conv1_t_stride=1, no_max_pool=True)


def normalize(img: np.ndarray) -> np.ndarray:
    """Per-image intensity normalisation shared by training, evaluation and serving.

    Clips to the 1st-99th percentile and rescales to [0, 1]. Because it is invariant to any affine change
    of intensity, a DICOM stored as 12/14/16-bit, a PNG and the training arrays all map to the same input.
    (An earlier version trained on raw /255 values but served min-max rescaled DICOMs: specificity fell
    from 0.99 to 0.00. tests/test_imaging.py::test_dicom_and_png_paths_agree guards this.)"""
    img = np.asarray(img, dtype="float32")
    lo, hi = np.percentile(img, (1, 99))
    return np.clip((img - lo) / max(float(hi - lo), 1e-6), 0, 1).astype("float32")


def preprocess(img: np.ndarray, size: int = 128) -> torch.Tensor:
    """(H,W) image of any intensity scale -> normalised (1,1,size,size) tensor."""
    t = torch.from_numpy(np.ascontiguousarray(normalize(img)))[None, None]
    if t.shape[-1] != size or t.shape[-2] != size:
        t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t


def load_model(path: str, device: str = "cpu") -> torch.nn.Module:
    m = build_model()
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device).eval()
