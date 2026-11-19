"""Train the classifier with MONAI transforms + DataLoader. python -m imaging.train"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, EnsureTyped, RandFlipd, RandGaussianNoised, RandRotated, RandScaleIntensityd
from sklearn.metrics import roc_auc_score

from .model import build_model, normalize


def load_split(d: Path, name: str):
    z = np.load(d / f"{name}.npz")
    images = np.stack([normalize(im) for im in z["images"].astype("float32")])  # same function as serving
    return images, z["labels"], z["lesions"]


def make_loader(images, labels, train: bool, batch: int):
    items = [{"img": im[None], "label": int(y)} for im, y in zip(images, labels)]
    tf = [EnsureTyped(keys="img", dtype=torch.float32)]
    if train:
        tf += [RandFlipd("img", prob=0.5, spatial_axis=1), RandRotated("img", range_x=0.15, prob=0.5, padding_mode="border"),
               RandScaleIntensityd("img", factors=0.1, prob=0.5), RandGaussianNoised("img", prob=0.3, std=0.01)]
    return DataLoader(Dataset(items, Compose(tf)), batch_size=batch, shuffle=train, num_workers=0)


@torch.no_grad()
def predict_proba(model, images, batch=128, device="cpu"):
    model.eval()
    out = []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(images[i:i + batch])[:, None].to(device)
        out.append(torch.softmax(model(x), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    d, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (xtr, ytr, _), (xva, yva, _) = load_split(d, "train"), load_split(d, "val")
    loader = make_loader(xtr, ytr, True, a.batch)

    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(loader))
    loss_fn = torch.nn.CrossEntropyLoss()

    best, history = -1.0, []
    for ep in range(a.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for b in loader:
            opt.zero_grad()
            loss = loss_fn(model(b["img"]), b["label"])
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b["label"])
        auc = roc_auc_score(yva, predict_proba(model, xva))
        history.append({"epoch": ep + 1, "train_loss": tot / len(xtr), "val_auc": float(auc), "seconds": time.time() - t0})
        print(history[-1], flush=True)
        if auc > best:
            best = auc
            torch.save(model.state_dict(), out / "model.pt")
    (out / "train_metrics.json").write_text(json.dumps({"best_val_auc": best, "history": history}, indent=2))


if __name__ == "__main__":
    main()
