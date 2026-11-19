"""Pre-compute results for the 6 sample scans so the portal can run as a static demo (GitHub Pages).

    PYTHONPATH=. python scripts/make_demo_assets.py

Runs the real model and Grad-CAM once and writes frontend/public/demo/{id}/image.png, heatmap.png and
frontend/public/demo/studies.json. Nothing is faked: these are the same outputs the live backend produces.
"""
import json
from pathlib import Path

import torch

from imaging.dicom_utils import read_dicom
from imaging.gradcam import grey_png, heatmap, overlay
from imaging.model import CLASSES, load_model, preprocess

out = Path("frontend/public/demo")
model = load_model("pretrained/model.pt")
studies = []
for k in range(6):
    raw = Path(f"data/samples/sample_{k}.dcm").read_bytes()
    img, meta = read_dicom(raw)
    x = preprocess(img)
    with torch.no_grad():
        prob = float(torch.softmax(model(x), 1)[0, 1])
    small = x[0, 0].numpy()
    d = out / str(k + 1)
    d.mkdir(parents=True, exist_ok=True)
    (d / "image.png").write_bytes(grey_png(small))
    (d / "heatmap.png").write_bytes(overlay(small, heatmap(model, x, class_idx=1)))
    studies.append({
        "id": k + 1, "filename": f"sample_{k}.dcm", "patient_id": meta["patient_id"], "patient_name": meta["patient_name"],
        "modality": meta["modality"], "orthanc_id": None, "created_at": None, "ai_label": CLASSES[int(prob >= 0.5)],
        "ai_prob": prob, "status": "pending_review", "review_decision": None, "final_label": None,
        "review_notes": None, "reviewer": None, "reviewed_at": None,
    })
(out / "studies.json").write_text(json.dumps(studies, indent=1))
for s in studies:
    print(s["id"], s["patient_id"], s["ai_label"], round(s["ai_prob"], 3))
