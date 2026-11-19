import numpy as np
import pytest

from recsys import index as faiss_index
from recsys.cache import DictStore, FeatureCache
from recsys.config import Config
from recsys.data_gen import generate


def tiny_cfg(tmp_path) -> Config:
    return Config(n_users=120, n_items=80, n_categories=5, n_countries=3, n_age_buckets=3, n_price_bins=4,
                  interactions_per_user=20, holdout_per_user=3, retrieval_epochs=1, ranking_epochs=1,
                  retrieval_batch=256, ranking_batch=256, n_candidates=30, top_k=5, index_kind="hnsw",
                  data_dir=str(tmp_path / "data"), artifact_dir=str(tmp_path / "art"))


def test_generate_shapes_and_ids(tmp_path):
    cfg = tiny_cfg(tmp_path)
    d = generate(cfg)
    assert len(d["users"]) == 120 and len(d["items"]) == 80
    inter = d["interactions"]
    assert len(inter) == 120 * 20
    assert not inter.duplicated(["user_id", "item_id"]).any()
    assert inter.item_id.between(0, 79).all()


def test_hnsw_matches_exact_search():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(500, 16)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    approx, exact = faiss_index.build_index(emb, "hnsw"), faiss_index.build_index(emb, "flat")
    assert faiss_index.overlap_with_exact(approx, exact, emb[:50], 20) > 0.95


def test_ivf_index_builds_and_searches():
    emb = np.random.default_rng(1).normal(size=(400, 8)).astype("float32")
    idx = faiss_index.build_index(emb, "ivf")
    _, ids = faiss_index.search(idx, emb[:3], 5)
    assert ids.shape == (3, 5)


def test_cache_roundtrip_in_memory():
    c = FeatureCache(DictStore())
    c.warm(np.eye(3, dtype="float32"), np.array([[1, 2, 0.5]] * 3, "float32"),
           np.array([[0, 1, 2.0], [3, 4, 1.0]], "float32"), {0: np.array([1, 2])})
    assert c.user_embedding(1).tolist() == [0, 1, 0]
    assert c.seen(0).tolist() == [1, 2] and c.seen(2).size == 0
    assert c.item_features([1, 0]).shape == (2, 3)
    assert c.user_embedding(99) is None
    c.set_recs(0, 5, [{"item_id": 1}])
    assert c.get_recs(0, 5) == [{"item_id": 1}]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from recsys import train
    from recsys.data_gen import write_raw
    from recsys.spark_etl import run

    cfg = tiny_cfg(tmp_path_factory.mktemp("rec"))
    write_raw(cfg)
    run(cfg)
    train.run_all(cfg)
    return cfg


def test_etl_split_is_temporal_and_disjoint(trained):
    import pandas as pd

    p = trained.processed_dir
    tr, va, te = (pd.read_parquet(p / n) for n in ("train", "val", "test"))
    assert te.groupby("user_id").size().eq(3).all() and va.groupby("user_id").size().eq(3).all()
    key = lambda d: set(zip(d.user_id, d.item_id))
    assert not (key(tr) & key(va)) and not (key(tr) & key(te)) and not (key(va) & key(te))
    # test is strictly the most recent per user
    assert (te.groupby("user_id").ts.min() >= va.groupby("user_id").ts.max()).all()


def test_two_stage_recommend(trained):
    from recsys.pipeline import Recommender

    rec = Recommender(trained, cache=FeatureCache(DictStore()))
    out = rec.recommend(7, 5, use_cache=False)
    scores = [r["score"] for r in out["items"]]
    assert len(scores) == 5 and scores == sorted(scores, reverse=True)
    assert not set(r["item_id"] for r in out["items"]) & set(rec.cache.seen(7).tolist())
    assert rec.recommend(7, 5)["cached"] is False and rec.recommend(7, 5)["cached"] is True
    with pytest.raises(KeyError):
        rec.recommend(10_000)


def test_api(trained, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("REC_ARTIFACT_DIR", str(trained.art))
    monkeypatch.setenv("REDIS_URL", "redis://localhost:1/0")  # unreachable -> memory fallback
    from serving.api import app

    with TestClient(app) as c:
        assert c.get("/health").json()["cache"] == "memory"
        r = c.get("/recommend/3?k=4")
        assert r.status_code == 200 and len(r.json()["items"]) == 4
        assert c.get("/recommend/99999").status_code == 404


def test_triton_export_config(trained, tmp_path, monkeypatch):
    from serving.triton import export

    monkeypatch.setattr(export, "REPO", tmp_path / "repo")
    dst = export.export(trained)
    cfg = (dst / "config.pbtxt").read_text()
    assert 'platform: "tensorflow_savedmodel"' in cfg and '"retr_score"' in cfg
    assert (dst / "1" / "model.savedmodel" / "saved_model.pb").exists()
