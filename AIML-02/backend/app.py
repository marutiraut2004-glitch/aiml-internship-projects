"""FastAPI backend.  uvicorn backend.app:app --port 8000

Env: IMG_MODEL_PATH (default: models/model.pt, else pretrained/model.pt), IMG_DATA_DIR (runtime), ORTHANC_URL / ORTHANC_USER / ORTHANC_PASSWORD (optional)
Not a medical device. No authentication: put it behind your own auth before any real use.
"""
from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field

from imaging.dicom_utils import is_dicom, read_dicom
from imaging.gradcam import grey_png, heatmap, overlay
from imaging.model import CLASSES, load_model, preprocess

from .db import DB
from .storage import orthanc_from_env

MAX_BYTES = 30 * 1024 * 1024
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    data = Path(os.environ.get("IMG_DATA_DIR", "runtime"))
    (data / "studies").mkdir(parents=True, exist_ok=True)
    state["dir"] = data
    state["db"] = DB(str(data / "worklist.db"))
    state["orthanc"] = orthanc_from_env()
    # your own run (models/, from `dvc repro`) wins over the bundled pretrained weights
    candidates = [os.environ["IMG_MODEL_PATH"]] if "IMG_MODEL_PATH" in os.environ else ["models/model.pt", "pretrained/model.pt"]
    model_path = next((c for c in candidates if Path(c).exists()), None)
    state["model"] = load_model(model_path) if model_path else None
    yield


app = FastAPI(title="AI-assisted imaging review", lifespan=lifespan)


def decode_image(data: bytes, filename: str):
    if is_dicom(data) or filename.lower().endswith((".dcm", ".dicom")):
        try:
            return read_dicom(data)
        except Exception as e:
            raise HTTPException(422, f"could not read DICOM: {e}")
    try:
        im = Image.open(io.BytesIO(data)).convert("L")
    except Exception:
        raise HTTPException(422, "file is neither DICOM nor a readable image")
    return np.asarray(im, dtype="float32") / 255.0, {"patient_id": "", "patient_name": "", "modality": "", "study_uid": ""}


@app.get("/api/health")
def health():
    o = state["orthanc"]
    return {"status": "ok", "model_loaded": state["model"] is not None,
            "orthanc": "disabled" if o is None else ("up" if o.available() else "unreachable")}


@app.post("/api/studies")
async def upload(file: UploadFile = File(...)):
    if state["model"] is None:
        raise HTTPException(503, "no trained model found; run `python -m imaging.train` first")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "file too large")
    img, meta = decode_image(data, file.filename or "")

    x = preprocess(img)
    with torch.no_grad():
        prob = float(torch.softmax(state["model"](x), 1)[0, 1])
    label = CLASSES[int(prob >= 0.5)]
    heat = heatmap(state["model"], x, class_idx=1)
    small = x[0, 0].numpy()

    orthanc_id = None
    if is_dicom(data) and state["orthanc"] is not None:
        orthanc_id = state["orthanc"].upload(data)

    sid = state["db"].add(filename=file.filename, patient_id=meta["patient_id"], patient_name=meta["patient_name"],
                          modality=meta["modality"], orthanc_id=orthanc_id, ai_label=label, ai_prob=prob)
    d = state["dir"] / "studies" / str(sid)
    d.mkdir(parents=True)
    (d / "image.png").write_bytes(grey_png(small))
    (d / "heatmap.png").write_bytes(overlay(small, heat))
    (d / "heat.npy").write_bytes(heat.astype("float16").tobytes())
    return state["db"].get(sid)


@app.get("/api/studies")
def list_studies(status: Literal["pending_review", "reviewed"] | None = None):
    return state["db"].list(status)


@app.get("/api/studies/{sid}")
def get_study(sid: int):
    s = state["db"].get(sid)
    if not s:
        raise HTTPException(404, "study not found")
    return s


@app.get("/api/studies/{sid}/{kind}")
def study_image(sid: int, kind: Literal["image", "heatmap"]):
    p = state["dir"] / "studies" / str(sid) / f"{kind}.png"
    if not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(p, media_type="image/png")


class Review(BaseModel):
    decision: Literal["agree", "disagree", "uncertain"]
    final_label: Literal["normal", "nodule"] | None = None
    notes: str = Field("", max_length=2000)
    reviewer: str = Field(..., min_length=1, max_length=100)


@app.post("/api/studies/{sid}/review")
def review(sid: int, r: Review):
    s = state["db"].get(sid)
    if not s:
        raise HTTPException(404, "study not found")
    final = r.final_label or (s["ai_label"] if r.decision == "agree" else None)
    if final is None and r.decision == "disagree":
        final = "normal" if s["ai_label"] == "nodule" else "nodule"  # the only other class
    state["db"].review(sid, r.decision, final or "", r.notes, r.reviewer)
    return state["db"].get(sid)
