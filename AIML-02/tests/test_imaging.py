import io
import json
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from imaging import synth
from imaging.dicom_utils import array_to_dicom, dicom_bytes, is_dicom, read_dicom
from imaging.gradcam import heatmap, overlay, pointing_hit
from imaging.model import build_model, preprocess


def test_synth_split_balanced_and_lesions_labelled():
    x, y, les = synth.make_split(40, seed=1)
    assert x.shape == (40, 128, 128) and x.dtype == np.uint8
    assert y.sum() == 20
    assert (les[y == 1, 2] > 0).all() and (les[y == 0] == -1).all()
    cx, cy = les[y == 1, 0], les[y == 1, 1]
    assert synth.LUNG_MASK[cy.astype(int), cx.astype(int)].all()  # lesions sit inside lung fields


def test_dicom_roundtrip():
    arr = (np.random.default_rng(0).random((128, 128)) * 255).astype("uint8")
    raw = dicom_bytes(array_to_dicom(arr, patient_id="P1"))
    assert is_dicom(raw) and not is_dicom(b"not dicom")
    out, meta = read_dicom(raw)
    assert meta["patient_id"] == "P1" and meta["modality"] == "DX"
    assert np.abs(out - arr / 255.0).max() < 1e-2


def test_normalize_is_affine_invariant():
    from imaging.model import normalize

    a = np.random.default_rng(0).random((64, 64)).astype("float32")
    assert np.abs(normalize(a) - normalize(a * 3000 + 17)).max() < 1e-3


def test_dicom_and_png_paths_agree():
    """The training array, a DICOM of it and a PNG of it must reach the model as the same tensor."""
    u8 = synth.make_image(np.random.default_rng(5), True)[0]
    u8 = (u8 * 255).round().astype("uint8")
    from_dicom = preprocess(read_dicom(dicom_bytes(array_to_dicom(u8)))[0])
    from_png = preprocess(np.asarray(Image.fromarray(u8), dtype="float32") / 255.0)
    from_array = preprocess(u8.astype("float32") / 255.0)
    assert (from_dicom - from_array).abs().max() < 0.02 and (from_png - from_array).abs().max() < 0.02


def test_preprocess_resizes():
    assert preprocess(np.zeros((300, 200), "float32")).shape == (1, 1, 128, 128)


def test_gradcam_shape_range_and_overlay():
    m = build_model().eval()
    x = preprocess(np.random.default_rng(0).random((128, 128)).astype("float32"))
    h = heatmap(m, x)
    assert h.shape == (128, 128) and 0 <= h.min() and h.max() <= 1
    png = overlay(x[0, 0].numpy(), h)
    assert Image.open(io.BytesIO(png)).size == (384, 384)


WEIGHTS = next((p for p in (Path("models/model.pt"), Path("pretrained/model.pt")) if p.exists()), None)


@pytest.mark.skipif(WEIGHTS is None, reason="needs trained weights (python -m imaging.train)")
def test_gradcam_localises_inserted_lesion():
    """Regression guard: put a known lesion into normal images, the heatmap peak must land on it."""
    from imaging.model import load_model

    m = load_model(str(WEIGHTS))
    rng = np.random.default_rng(1)
    x, y, _ = synth.make_split(60, seed=11)
    hits = 0
    negs = [im for im, lab in zip(x, y) if lab == 0][:25]
    for im in negs:
        cx, cy = synth._random_inside(rng, 16)
        img = np.clip(im / 255.0 + synth._blob(cx, cy, 7, 0.15), 0, 1).astype("float32")
        hits += pointing_hit(heatmap(m, preprocess(img)), (cx, cy, 7))
    assert hits / len(negs) >= 0.8


def test_pointing_hit():
    h = np.zeros((128, 128))
    h[50, 60] = 1
    assert pointing_hit(h, (60, 50, 5)) and not pointing_hit(h, (100, 100, 5))


# ---------------------------------------------------------------- backend
class FakeOrthanc(BaseHTTPRequestHandler):
    received = []

    def log_message(self, *a): pass

    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"{}")

    def do_POST(self):
        FakeOrthanc.received.append(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200); self.end_headers(); self.wfile.write(json.dumps({"ID": "abc-123"}).encode())


@pytest.fixture()
def client(tmp_path, monkeypatch):
    torch.manual_seed(0)
    torch.save(build_model().state_dict(), tmp_path / "model.pt")  # untrained weights: tests plumbing only
    srv = HTTPServer(("127.0.0.1", 0), FakeOrthanc)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    FakeOrthanc.received.clear()
    monkeypatch.setenv("IMG_MODEL_PATH", str(tmp_path / "model.pt"))
    monkeypatch.setenv("IMG_DATA_DIR", str(tmp_path / "rt"))
    monkeypatch.setenv("ORTHANC_URL", f"http://127.0.0.1:{srv.server_port}")
    from backend.app import app

    with TestClient(app) as c:
        yield c
    srv.shutdown()


def _dicom():
    arr = synth.make_image(np.random.default_rng(3), True)[0]
    return dicom_bytes(array_to_dicom((arr * 255).astype("uint8"), patient_id="PT-9"))


def test_upload_dicom_runs_model_and_pushes_to_orthanc(client):
    assert client.get("/api/health").json() == {"status": "ok", "model_loaded": True, "orthanc": "up"}
    r = client.post("/api/studies", files={"file": ("a.dcm", _dicom())})
    assert r.status_code == 200
    s = r.json()
    assert s["patient_id"] == "PT-9" and s["orthanc_id"] == "abc-123" and s["status"] == "pending_review"
    assert s["ai_label"] in ("normal", "nodule") and 0 <= s["ai_prob"] <= 1
    assert len(FakeOrthanc.received) == 1 and is_dicom(FakeOrthanc.received[0])
    for kind in ("image", "heatmap"):
        assert client.get(f"/api/studies/{s['id']}/{kind}").headers["content-type"] == "image/png"


def test_upload_png_and_rejects_garbage(client):
    buf = io.BytesIO()
    Image.fromarray((np.random.default_rng(0).random((64, 64)) * 255).astype("uint8")).save(buf, "PNG")
    r = client.post("/api/studies", files={"file": ("x.png", buf.getvalue())})
    assert r.status_code == 200 and r.json()["orthanc_id"] is None  # only DICOM goes to Orthanc
    assert client.post("/api/studies", files={"file": ("x.txt", b"hello")}).status_code == 422
    assert client.post("/api/studies", files={"file": ("x.dcm", b"hello")}).status_code == 422


def test_review_flow(client):
    sid = client.post("/api/studies", files={"file": ("a.dcm", _dicom())}).json()["id"]
    ai = client.get(f"/api/studies/{sid}").json()["ai_label"]
    assert len(client.get("/api/studies?status=pending_review").json()) == 1

    r = client.post(f"/api/studies/{sid}/review", json={"decision": "disagree", "notes": "artefact", "reviewer": "Dr A"})
    assert r.status_code == 200
    s = r.json()
    assert s["status"] == "reviewed" and s["reviewer"] == "Dr A"
    assert s["final_label"] == ("normal" if ai == "nodule" else "nodule")  # disagree flips the AI label
    assert client.get("/api/studies?status=pending_review").json() == []

    assert client.post(f"/api/studies/{sid}/review", json={"decision": "maybe", "reviewer": "x"}).status_code == 422
    assert client.post(f"/api/studies/{sid}/review", json={"decision": "agree"}).status_code == 422  # reviewer required
    assert client.post("/api/studies/999/review", json={"decision": "agree", "reviewer": "x"}).status_code == 404
    assert client.get("/api/studies/999").status_code == 404


def test_503_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("IMG_MODEL_PATH", str(tmp_path / "missing.pt"))
    monkeypatch.setenv("IMG_DATA_DIR", str(tmp_path / "rt"))
    monkeypatch.delenv("ORTHANC_URL", raising=False)
    from backend.app import app

    with TestClient(app) as c:
        assert c.get("/api/health").json()["model_loaded"] is False
        assert c.post("/api/studies", files={"file": ("a.dcm", _dicom())}).status_code == 503
