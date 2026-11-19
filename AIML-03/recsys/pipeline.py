"""Online two-stage recommender: FAISS retrieval, Redis features, ranker (local or Triton)."""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from . import index as faiss_index
from .cache import FeatureCache
from .config import Config
from .models import RANKER_INPUTS


class LocalRanker:
    """Loads the exported SavedModel in-process."""

    def __init__(self, path: str):
        import tensorflow as tf

        self._tf = tf
        self.fn = tf.saved_model.load(path).signatures["serving_default"]

    def score(self, cols: dict[str, np.ndarray]) -> np.ndarray:
        tf = self._tf
        feeds = {n: tf.constant(cols[n].astype(dt.as_numpy_dtype)) for n, dt in RANKER_INPUTS.items()}
        return self.fn(**feeds)["score"].numpy()


class TritonRanker:
    """Calls a Triton Inference Server over HTTP. See serving/triton/README.md."""

    def __init__(self, url: str, model: str = "ranker"):
        import tritonclient.http as httpclient

        self.http, self.model = httpclient, model
        self.client = httpclient.InferenceServerClient(url=url)

    def score(self, cols: dict[str, np.ndarray]) -> np.ndarray:
        inputs = []
        for name, dt in RANKER_INPUTS.items():
            arr = cols[name].astype(dt.as_numpy_dtype)
            inp = self.http.InferInput(name, list(arr.shape), "INT32" if arr.dtype == np.int32 else "FP32")
            inp.set_data_from_numpy(arr)
            inputs.append(inp)
        res = self.client.infer(self.model, inputs, outputs=[self.http.InferRequestedOutput("score")])
        return res.as_numpy("score")


class Recommender:
    def __init__(self, cfg: Config | None = None, cache: FeatureCache | None = None, ranker_backend: str | None = None):
        self.cfg = cfg or Config.load()
        art = self.cfg.art
        self.item_emb = np.load(art / "item_emb.npy")
        user_emb = np.load(art / "user_emb.npy")

        p = self.cfg.processed_dir
        users = pd.read_parquet(p / "user_features").sort_values("user_id")
        items = pd.read_parquet(p / "item_features").sort_values("item_id")
        train = pd.concat([pd.read_parquet(p / "train"), pd.read_parquet(p / "val")])
        seen = {int(u): g.item_id.to_numpy() for u, g in train.groupby("user_id")}

        self.index = faiss_index.build_index(self.item_emb, self.cfg.index_kind)
        self.exact = faiss_index.build_index(self.item_emb, "flat")

        self.cache = cache or FeatureCache(rec_ttl=self.cfg.rec_cache_ttl)
        self.cache.warm(
            user_emb,
            users[["age_bucket", "country", "user_activity_log"]].to_numpy("float32"),
            items[["category", "price_bin", "item_pop_log"]].to_numpy("float32"),
            seen,
        )
        self.user_emb = user_emb  # kept only for offline evaluation helpers

        backend = ranker_backend or os.environ.get("RANKER_BACKEND", "local")
        if backend == "triton":
            self.ranker = TritonRanker(os.environ.get("TRITON_URL", "localhost:8000"))
        else:
            self.ranker = LocalRanker(str(art / "ranker_savedmodel"))
        self.backend = backend

    # ---- stage 1 ---------------------------------------------------------------
    def retrieve(self, uid: int, n: int | None = None):
        n = n or self.cfg.n_candidates
        emb = self.cache.user_embedding(uid)
        if emb is None:
            raise KeyError(f"unknown user {uid}")
        seen = self.cache.seen(uid)
        # over-fetch so that removing already-seen items still leaves n candidates
        scores, ids = faiss_index.search(self.index, emb, min(n + len(seen), len(self.item_emb)))
        ids, scores = ids[0], scores[0]
        keep = (ids >= 0) & ~np.isin(ids, seen)
        return ids[keep][:n], scores[keep][:n]

    # ---- stage 2 ---------------------------------------------------------------
    def rank(self, uid: int, cand: np.ndarray, ret_scores: np.ndarray) -> np.ndarray:
        uf = self.cache.user_features(uid)
        itf = self.cache.item_features(cand)
        m = len(cand)
        cols = {
            "user_id": np.full(m, uid), "age_bucket": np.full(m, int(uf[0])), "country": np.full(m, int(uf[1])),
            "item_id": cand, "category": itf[:, 0].astype(int), "price_bin": itf[:, 1].astype(int),
            "item_pop_log": itf[:, 2], "user_activity_log": np.full(m, uf[2], dtype="float32"),
            "retr_score": ret_scores.astype("float32"),
        }
        return self.ranker.score(cols)

    def recommend(self, uid: int, k: int | None = None, use_cache: bool = True) -> dict:
        k = k or self.cfg.top_k
        t0 = time.perf_counter()
        if use_cache:
            hit = self.cache.get_recs(uid, k)
            if hit is not None:
                return {"user_id": uid, "items": hit, "cached": True,
                        "latency_ms": (time.perf_counter() - t0) * 1000}
        cand, ret_scores = self.retrieve(uid)
        t1 = time.perf_counter()
        rank_scores = self.rank(uid, cand, ret_scores)
        t2 = time.perf_counter()
        order = np.argsort(-rank_scores)[:k]
        recs = [{"item_id": int(cand[i]), "score": float(rank_scores[i]), "retrieval_score": float(ret_scores[i])}
                for i in order]
        if use_cache:
            self.cache.set_recs(uid, k, recs)
        return {"user_id": uid, "items": recs, "cached": False,
                "latency_ms": (t2 - t0) * 1000,
                "retrieval_ms": (t1 - t0) * 1000, "ranking_ms": (t2 - t1) * 1000}
