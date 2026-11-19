"""Synthetic chest-X-ray-like images with known lesion locations.

No real patient data is used. Each 128x128 image has two lung fields, rib texture, noise and a few
small bright distractor spots (like vessels). Positive images also contain one low-contrast
nodule. Because we know where the nodule is, we can measure whether Grad-CAM points at it.

This is a stand-in so the whole platform can run and be tested. Swap in a real dataset
(e.g. a public chest X-ray set) by replacing `data/*.npz`, see README.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

SIZE = 128
LUNGS = ((42, 64), (86, 64))  # (cx, cy)
LUNG_AX = (20, 36)


def _lung_mask() -> np.ndarray:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    m = np.zeros((SIZE, SIZE), bool)
    for cx, cy in LUNGS:
        m |= ((xx - cx) / LUNG_AX[0]) ** 2 + ((yy - cy) / LUNG_AX[1]) ** 2 < 1
    return m


LUNG_MASK = _lung_mask()
_YY, _XX = np.mgrid[0:SIZE, 0:SIZE]


def _blob(cx, cy, r, amp):
    return amp * np.exp(-((_XX - cx) ** 2 + (_YY - cy) ** 2) / (2 * (r / 2) ** 2))


def _random_inside(rng, margin=10):
    inner = LUNG_MASK.copy()
    inner[:margin], inner[-margin:], inner[:, :margin], inner[:, -margin:] = False, False, False, False
    ys, xs = np.nonzero(inner)
    i = rng.integers(len(ys))
    return int(xs[i]), int(ys[i])


def make_image(rng: np.random.Generator, positive: bool):
    img = np.full((SIZE, SIZE), 0.55) + rng.normal(0, 0.02)
    img[LUNG_MASK] -= 0.22
    img += 0.03 * np.sin(_YY / rng.uniform(2.5, 3.5)) * LUNG_MASK  # ribs
    img += 0.05 * np.exp(-((_XX - 64) / 10) ** 2)  # mediastinum
    for _ in range(rng.integers(2, 6)):  # distractors present in every image
        cx, cy = _random_inside(rng)
        img += _blob(cx, cy, rng.uniform(2, 3.5), rng.uniform(0.06, 0.16))
    lesion = None
    if positive:
        cx, cy = _random_inside(rng, margin=14)
        r = float(rng.uniform(5, 9))
        img += _blob(cx, cy, r, rng.uniform(0.07, 0.22))
        lesion = (cx, cy, r)
    img = gaussian_filter(img, 1.0) + rng.normal(0, 0.03, img.shape)
    return np.clip(img, 0, 1).astype("float32"), lesion


def make_split(n: int, seed: int):
    rng = np.random.default_rng(seed)
    labels = (np.arange(n) % 2).astype("int64")
    rng.shuffle(labels)
    imgs = np.zeros((n, SIZE, SIZE), "uint8")
    lesions = np.full((n, 3), -1.0, "float32")
    for i, y in enumerate(labels):
        im, les = make_image(rng, bool(y))
        imgs[i] = (im * 255).round().astype("uint8")
        if les:
            lesions[i] = les
    return imgs, labels, lesions


def write_dataset(out: str | Path, n_train=3000, n_val=500, n_test=500, seed=0) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, n, s in (("train", n_train, seed), ("val", n_val, seed + 1), ("test", n_test, seed + 2)):
        x, y, les = make_split(n, s)
        np.savez_compressed(out / f"{name}.npz", images=x, labels=y, lesions=les)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    write_dataset(a.out, a.n_train, a.n_val, a.n_test, a.seed)

    from .dicom_utils import write_samples

    write_samples(a.out, Path(a.out) / "samples")
