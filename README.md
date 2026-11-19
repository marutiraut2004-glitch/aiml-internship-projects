# AI / ML Specialization Track: Projects AIML-03 and AIML-02

| Folder | Project | Stack |
|---|---|---|
| [`AIML-03/`](AIML-03) | Large-Scale Personalized Recommendation Engine | TensorFlow Recommenders, FAISS, Redis, Apache Spark, Airflow, Triton |
| [`AIML-02/`](AIML-02) | AI-Assisted Medical Imaging and Diagnosis System | PyTorch, MONAI, Grad-CAM, Orthanc (DICOM), DVC, Docker, React |

Each folder is self-contained, with its own README (architecture, run steps, results), tests and Docker files.

**Live demo:** https://marutiraut2004-glitch.github.io/aiml-internship-projects/ (static demo of the AIML-02 portal with precomputed results; see `AIML-02/README.md`).

**Looking for where something is in the code?** See [`CODE_MAP.md`](CODE_MAP.md).

## Quick start

```bash
# AIML-03: recommender (about 1 minute on CPU)
cd AIML-03 && pip install -r requirements.txt && python -m recsys.cli all && pytest -q

# AIML-02: imaging platform
cd AIML-02 && pip install -r requirements.txt
python -m imaging.synth && python -m imaging.train && python -m imaging.evaluate && pytest -q   # or just use the bundled weights
uvicorn backend.app:app --port 8000 &           # API
cd frontend && npm install && npm run dev        # UI on http://localhost:5173
```

## Honest scope notes

Both projects run end to end on **synthetic data** because real datasets and clinical images cannot be
shipped in a repository. Each README says exactly what was run and verified here and what was written but
not exercised against live infrastructure (Triton, Airflow scheduling, Docker builds, a live Orthanc server).
Metrics on synthetic data show the pipelines are correct; they are not evidence of real-world performance.
The imaging system is a research prototype, not a medical device.
