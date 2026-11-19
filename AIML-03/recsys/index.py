"""FAISS wrapper. Embeddings are L2 normalised, so inner product equals cosine similarity."""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


def build_index(emb: np.ndarray, kind: str = "hnsw", m: int = 32, ef_search: int = 128, nlist: int | None = None):
    emb = np.ascontiguousarray(emb, dtype="float32")
    d = emb.shape[1]
    if kind == "flat":
        index = faiss.IndexFlatIP(d)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = ef_search
    elif kind == "ivf":
        nlist = nlist or max(4, int(np.sqrt(len(emb))))
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(emb)
        index.nprobe = max(1, nlist // 4)
    else:
        raise ValueError(f"unknown index kind: {kind}")
    index.add(emb)
    return index


def search(index, queries: np.ndarray, k: int):
    q = np.ascontiguousarray(np.atleast_2d(queries), dtype="float32")
    scores, ids = index.search(q, k)
    return scores, ids


def save(index, path: str | Path) -> None:
    faiss.write_index(index, str(path))


def load(path: str | Path):
    return faiss.read_index(str(path))


def overlap_with_exact(index, exact, queries: np.ndarray, k: int) -> float:
    """Share of the exact top-k that the approximate index also returns."""
    _, a = search(index, queries, k)
    _, e = search(exact, queries, k)
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a, e)]))
