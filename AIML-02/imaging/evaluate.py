"""Test-set metrics plus Grad-CAM localisation. python -m imaging.evaluate"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .gradcam import heatmap, pointing_hit
from .model import load_model, preprocess
from .train import load_split, predict_proba


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="models/model.pt")
    ap.add_argument("--out", default="models/metrics.json")
    ap.add_argument("--cam-samples", type=int, default=100)
    a = ap.parse_args(argv)

    x, y, les = load_split(Path(a.data), "test")
    model = load_model(a.model)
    p = predict_proba(model, x)
    pred = p >= 0.5
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())

    pos = np.nonzero((y == 1) & pred)[0][: a.cam_samples]  # correctly detected positives
    hits = [pointing_hit(heatmap(model, preprocess(x[i])), les[i]) for i in pos]
    # chance level: the same test with the heatmap peak placed uniformly at random in the image
    rng = np.random.default_rng(0)
    chance = np.mean([pointing_hit(np.eye(128)[rng.integers(128)][:, None] * np.eye(128)[rng.integers(128)][None], les[i])
                      for i in pos for _ in range(20)]) if len(pos) else float("nan")

    res = {
        "n_test": int(len(y)), "auc": float(roc_auc_score(y, p)), "accuracy": float((pred == y).mean()),
        "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "gradcam_pointing_accuracy": float(np.mean(hits)) if hits else None,
        "gradcam_pointing_random_baseline": float(chance),
        "gradcam_samples": int(len(hits)),
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
