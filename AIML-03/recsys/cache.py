"""Redis-backed feature and recommendation cache, with an in-memory fallback.

Key layout (all values are small packed float32 blobs, fetched with one MGET per request):
    uemb:{user}   user embedding from the user tower
    ufeat:{user}  [age_bucket, country, user_activity_log]
    ifeat:{item}  [category, price_bin, item_pop_log]
    seen:{user}   int32 ids the user already interacted with in training
    rec:{user}:{k} JSON list of final recommendations, with TTL
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np

log = logging.getLogger(__name__)


class DictStore:
    """Same interface as RedisStore, used when no Redis server is reachable."""

    kind = "memory"

    def __init__(self):
        self.d: dict[str, bytes] = {}

    def mget(self, keys):
        return [self.d.get(k) for k in keys]

    def get(self, key):
        return self.d.get(key)

    def mset(self, mapping, ex=None):
        self.d.update(mapping)

    def set(self, key, value, ex=None):
        self.d[key] = value

    def flush(self):
        self.d.clear()


class RedisStore:
    kind = "redis"

    def __init__(self, url: str):
        import redis

        self.r = redis.Redis.from_url(url)
        self.r.ping()

    def mget(self, keys):
        return self.r.mget(keys) if keys else []

    def get(self, key):
        return self.r.get(key)

    def mset(self, mapping, ex=None):
        pipe = self.r.pipeline(transaction=False)
        for i, (k, v) in enumerate(mapping.items()):
            pipe.set(k, v, ex=ex)
            if i % 5000 == 4999:
                pipe.execute()
        pipe.execute()

    def set(self, key, value, ex=None):
        self.r.set(key, value, ex=ex)

    def flush(self):
        self.r.flushdb()


def make_store(url: str | None = None):
    url = url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    try:
        return RedisStore(url)
    except Exception as e:  # connection refused, redis package missing, ...
        log.warning("Redis unavailable (%s), using in-memory store", e)
        return DictStore()


def _f32(a) -> bytes:
    return np.asarray(a, dtype="float32").tobytes()


class FeatureCache:
    def __init__(self, store=None, rec_ttl: int = 60):
        self.store = store or make_store()
        self.rec_ttl = rec_ttl

    # ---- offline -> online load -------------------------------------------------
    def warm(self, user_emb, user_feats, item_feats, seen: dict[int, np.ndarray]) -> None:
        """user_feats: (n,3) float [age, country, activity_log]; item_feats: (m,3) [cat, price_bin, pop_log]."""
        m = {}
        for u in range(len(user_emb)):
            m[f"uemb:{u}"] = _f32(user_emb[u])
            m[f"ufeat:{u}"] = _f32(user_feats[u])
            m[f"seen:{u}"] = np.asarray(seen.get(u, []), dtype="int32").tobytes()
        for i in range(len(item_feats)):
            m[f"ifeat:{i}"] = _f32(item_feats[i])
        self.store.mset(m)

    # ---- online reads -----------------------------------------------------------
    def user_embedding(self, uid: int) -> np.ndarray | None:
        b = self.store.get(f"uemb:{uid}")
        return None if b is None else np.frombuffer(b, dtype="float32")

    def user_features(self, uid: int) -> np.ndarray | None:
        b = self.store.get(f"ufeat:{uid}")
        return None if b is None else np.frombuffer(b, dtype="float32")

    def seen(self, uid: int) -> np.ndarray:
        b = self.store.get(f"seen:{uid}")
        return np.empty(0, dtype="int32") if not b else np.frombuffer(b, dtype="int32")

    def item_features(self, item_ids) -> np.ndarray:
        blobs = self.store.mget([f"ifeat:{int(i)}" for i in item_ids])
        if any(b is None for b in blobs):
            raise KeyError("missing item features in cache")
        return np.stack([np.frombuffer(b, dtype="float32") for b in blobs])

    # ---- result cache -----------------------------------------------------------
    def get_recs(self, uid: int, k: int):
        b = self.store.get(f"rec:{uid}:{k}")
        return None if b is None else json.loads(b)

    def set_recs(self, uid: int, k: int, recs) -> None:
        self.store.set(f"rec:{uid}:{k}", json.dumps(recs).encode(), ex=self.rec_ttl)
