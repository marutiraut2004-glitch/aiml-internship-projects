"""Offline evaluation on the held-out most recent interactions, plus latency measurement."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import index as faiss_index
from .config import Config
from .pipeline import Recommender


def _metrics(ranked: np.ndarray, truth: set, k: int):
    hits = [1.0 if i in truth else 0.0 for i in ranked[:k]]
    dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(truth), k)))
    return sum(hits) / max(len(truth), 1), (dcg / idcg if idcg else 0.0), float(sum(hits) > 0)


def evaluate(cfg: Config, n_users: int = 1000, n_latency: int = 300, rec: Recommender | None = None) -> dict:
    rec = rec or Recommender(cfg)
    test = pd.read_parquet(cfg.processed_dir / "test")
    truth = {int(u): set(g.item_id) for u, g in test.groupby("user_id")}
    items = pd.read_parquet(cfg.processed_dir / "item_features").sort_values("item_id")
    pop_order = items.sort_values("item_pop", ascending=False).item_id.to_numpy()

    rng = np.random.default_rng(0)
    users = rng.choice(sorted(truth), size=min(n_users, len(truth)), replace=False)
    K, N = cfg.top_k, cfg.n_candidates

    agg = {n: [] for n in ["pop", "retrieval_only", "two_stage"]}
    cand_recall = []
    for u in users:
        u = int(u)
        t = truth[u]
        cand, cand_scores = rec.retrieve(u)
        cand_recall.append(len(t & set(cand)) / len(t))
        seen = set(rec.cache.seen(u).tolist())
        agg["pop"].append(_metrics(np.array([i for i in pop_order if i not in seen][:K]), t, K))
        agg["retrieval_only"].append(_metrics(cand, t, K))
        scores = rec.rank(u, cand, cand_scores)
        agg["two_stage"].append(_metrics(cand[np.argsort(-scores)], t, K))

    result = {
        "users_evaluated": len(users),
        f"candidate_recall@{N}": float(np.mean(cand_recall)),
        "ann_overlap_with_exact@200": faiss_index.overlap_with_exact(
            rec.index, rec.exact, rec.user_emb[users], min(200, len(rec.item_emb))),
    }
    for name, rows in agg.items():
        r = np.array(rows).mean(axis=0)
        result[name] = {f"recall@{K}": float(r[0]), f"ndcg@{K}": float(r[1]), f"hit_rate@{K}": float(r[2])}

    # latency: cold (no result cache) end to end, then cached
    lat_users = rng.choice(len(rec.user_emb), size=min(n_latency, len(rec.user_emb)), replace=False)
    rec.recommend(int(lat_users[0]), use_cache=False)  # warm up TF
    cold = [rec.recommend(int(u), use_cache=False) for u in lat_users]
    for u in lat_users:
        rec.recommend(int(u))
    cached = [rec.recommend(int(u)) for u in lat_users]
    pct = lambda xs, q: float(np.percentile(xs, q))
    result["latency_ms"] = {
        "cache_backend": rec.cache.store.kind, "ranker_backend": rec.backend, "index": cfg.index_kind,
        "uncached_p50": pct([r["latency_ms"] for r in cold], 50),
        "uncached_p95": pct([r["latency_ms"] for r in cold], 95),
        "retrieval_p50": pct([r["retrieval_ms"] for r in cold], 50),
        "ranking_p50": pct([r["ranking_ms"] for r in cold], 50),
        "cached_p50": pct([r["latency_ms"] for r in cached], 50),
    }
    (cfg.art / "metrics.json").write_text(json.dumps(result, indent=2))
    return result
