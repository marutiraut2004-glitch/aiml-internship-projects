"""FastAPI service.  uvicorn serving.api:app --port 8080

RANKER_BACKEND=local|triton, REDIS_URL, TRITON_URL, REC_ARTIFACT_DIR, REC_DATA_DIR are read from the environment.
"""
from __future__ import annotations

import statistics
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from recsys.pipeline import Recommender

state: dict = {"rec": None, "lat": deque(maxlen=1000)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["rec"] = Recommender()
    yield


app = FastAPI(title="Two-stage recommender", lifespan=lifespan)


@app.get("/health")
def health():
    rec = state["rec"]
    return {"status": "ok", "cache": rec.cache.store.kind, "ranker": rec.backend, "index": rec.cfg.index_kind}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = Query(10, ge=1, le=100), cache: bool = True):
    try:
        out = state["rec"].recommend(user_id, k, use_cache=cache)
    except KeyError:
        raise HTTPException(404, f"unknown user {user_id}")
    state["lat"].append(out["latency_ms"])
    return out


@app.get("/stats")
def stats():
    lat = sorted(state["lat"])
    if not lat:
        return {"requests": 0}
    return {"requests": len(lat), "p50_ms": statistics.median(lat), "p95_ms": lat[int(0.95 * (len(lat) - 1))]}
