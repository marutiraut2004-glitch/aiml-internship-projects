# Triton Inference Server for the ranker

```bash
python -m recsys.cli export-triton        # writes serving/triton/model_repository/ranker/{config.pbtxt,1/model.savedmodel}
docker compose --profile triton up triton # serves HTTP on :8000
RANKER_BACKEND=triton TRITON_URL=localhost:8000 uvicorn serving.api:app --port 8080
```

Status: the export script and the HTTP client (`recsys/pipeline.py::TritonRanker`) are written, but
**they were not run against a live Triton server** while building this project (no GPU or Triton image
in the build sandbox). The SavedModel they export is loaded and tested through the local backend.

Note that NVIDIA removed the TensorFlow backend from recent Triton releases. Use a container tag that
still ships it (the compose file pins `24.08-tf2-py3`, the last series with `tensorflow_savedmodel`),
or convert the model to ONNX and switch `platform` to `onnxruntime_onnx`.
