"""Read and write DICOM. Only what the platform needs: single-frame greyscale images."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

DX_STORAGE = "1.2.840.10008.5.1.4.1.1.1.1"  # Digital X-Ray Image Storage


def is_dicom(data: bytes) -> bool:
    return len(data) > 132 and data[128:132] == b"DICM"


def array_to_dicom(arr_u8: np.ndarray, patient_id="TEST-0001", patient_name="Synthetic^Patient") -> FileDataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = DX_STORAGE
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = DX_STORAGE, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = generate_uid(), generate_uid()
    ds.PatientID, ds.PatientName = patient_id, patient_name
    ds.Modality, ds.StudyDescription = "DX", "Synthetic chest X-ray"
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.Rows, ds.Columns = arr_u8.shape
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PixelData = (arr_u8.astype("uint16") * 257).tobytes()  # scale 0..255 -> 0..65535
    return ds


def dicom_bytes(ds: FileDataset) -> bytes:
    buf = io.BytesIO()
    ds.save_as(buf)
    return buf.getvalue()


def read_dicom(data: bytes):
    """Return (float32 image in [0,1], metadata dict)."""
    ds = pydicom.dcmread(io.BytesIO(data))
    arr = ds.pixel_array.astype("float32")
    if arr.ndim == 3:
        arr = arr[0]
    arr = arr * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = 1.0 - arr
    meta = {"patient_id": str(getattr(ds, "PatientID", "")), "patient_name": str(getattr(ds, "PatientName", "")),
            "modality": str(getattr(ds, "Modality", "")), "study_uid": str(getattr(ds, "StudyInstanceUID", ""))}
    return arr, meta


def write_samples(dataset_dir: str | Path, out: str | Path, n: int = 6) -> list[Path]:
    """A few DICOM files (half abnormal) for demos and for uploading to Orthanc."""
    d = np.load(Path(dataset_dir) / "test.npz")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, i in enumerate(list(np.nonzero(d["labels"] == 1)[0][: n // 2]) + list(np.nonzero(d["labels"] == 0)[0][: n // 2])):
        ds = array_to_dicom(d["images"][i], patient_id=f"DEMO-{k:04d}", patient_name=f"Demo^Patient{k}")
        p = out / f"sample_{k}.dcm"
        ds.save_as(p)
        paths.append(p)
    return paths
