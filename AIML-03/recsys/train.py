"""Training for both stages. Reads the parquet tables written by the Spark ETL."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import Config
from .models import RANKER_INPUTS, RankerServing, RankingModel, TwoTower

SEED_OPS = 0


def load_tables(cfg: Config):
    p = cfg.processed_dir
    train = pd.read_parquet(p / "train")
    val = pd.read_parquet(p / "val")
    users = pd.read_parquet(p / "user_features").sort_values("user_id").reset_index(drop=True)
    items = pd.read_parquet(p / "item_features").sort_values("item_id").reset_index(drop=True)
    assert (users.user_id.values == np.arange(len(users))).all(), "user ids must be 0..n-1"
    assert (items.item_id.values == np.arange(len(items))).all(), "item ids must be 0..n-1"
    return train, val, users, items


def _to_ds(df: pd.DataFrame, cols: dict, batch: int, seed: int) -> tf.data.Dataset:
    data = {c: df[c].to_numpy().astype(dt) for c, dt in cols.items()}
    ds = tf.data.Dataset.from_tensor_slices(data)
    return ds.shuffle(min(len(df), 200_000), seed=seed).batch(batch).prefetch(tf.data.AUTOTUNE)


def train_retrieval(cfg: Config) -> dict:
    tf.keras.utils.set_random_seed(cfg.seed)
    train, _, users, items = load_tables(cfg)
    df = train.merge(users, on="user_id").merge(items, on="item_id")
    prob = (items.item_pop.clip(lower=1) / items.item_pop.sum()).to_numpy()
    df["item_prob"] = prob[df.item_id.to_numpy()]

    cols = {"user_id": np.int32, "age_bucket": np.int32, "country": np.int32, "item_id": np.int32,
            "category": np.int32, "price_bin": np.int32, "item_prob": np.float32}
    ds = _to_ds(df, cols, cfg.retrieval_batch, cfg.seed)

    model = TwoTower(cfg)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.003))
    t0 = time.time()
    hist = model.fit(ds, epochs=cfg.retrieval_epochs, verbose=2)

    def embed(tower, table, id_col, cols_):
        feats = {c: tf.constant(table[c].to_numpy().astype(np.int32)) for c in cols_}
        return tower(feats).numpy().astype("float32")

    user_emb = embed(model.user_tower, users, "user_id", ["user_id", "age_bucket", "country"])
    item_emb = embed(model.item_tower, items, "item_id", ["item_id", "category", "price_bin"])
    cfg.art.mkdir(parents=True, exist_ok=True)
    np.save(cfg.art / "user_emb.npy", user_emb)
    np.save(cfg.art / "item_emb.npy", item_emb)
    return {"retrieval_loss": [float(x) for x in hist.history["loss"]], "retrieval_seconds": time.time() - t0}


def build_ranker_examples(cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    """Ranker training rows built the way serving will see them.

    For every user, take the retrieval model's top-N candidates (excluding items in `train`).
    Items the user interacted with in `val` are positives, the rest of the candidates are
    hard negatives (a fixed number sampled per user). Val positives that retrieval missed
    are kept too, with their retrieval score computed directly. Because the retrieval model
    never trained on `val`, the retrieval score distribution matches serving.
    """
    train, val, users, items = load_tables(cfg)
    user_emb = np.load(cfg.art / "user_emb.npy")
    item_emb = np.load(cfg.art / "item_emb.npy")
    scores = user_emb @ item_emb.T
    scores[train.user_id.to_numpy(), train.item_id.to_numpy()] = -np.inf  # drop train-seen items
    n = min(cfg.n_candidates, cfg.n_items - 1)
    cand = np.argpartition(-scores, n, axis=1)[:, :n]

    pos = {int(u): set(g.item_id) for u, g in val.groupby("user_id")}
    rows = []
    for u in range(cfg.n_users):
        p = pos.get(u, set())
        c = cand[u]
        negs = [i for i in c if i not in p]
        negs = rng.choice(negs, size=min(cfg.neg_per_pos * cfg.holdout_per_user * 2, len(negs)), replace=False)
        for i in p:
            rows.append((u, i, 1.0))
        for i in negs:
            rows.append((u, int(i), 0.0))
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "label"])
    df["retr_score"] = (user_emb[df.user_id.to_numpy()] * item_emb[df.item_id.to_numpy()]).sum(1).astype("float32")
    return df.merge(users, on="user_id").merge(items, on="item_id")


def train_ranker(cfg: Config) -> dict:
    tf.keras.utils.set_random_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    df = build_ranker_examples(cfg, rng)

    cols = {**{k: v.as_numpy_dtype for k, v in RANKER_INPUTS.items()}, "label": np.float32}
    ds = _to_ds(df, cols, cfg.ranking_batch, cfg.seed)

    model = RankingModel(cfg)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.003))
    t0 = time.time()
    hist = model.fit(ds, epochs=cfg.ranking_epochs, verbose=2)

    out = cfg.art / "ranker_savedmodel"
    serving = RankerServing(model.net)
    tf.saved_model.save(serving, str(out), signatures={"serving_default": serving.serve})
    return {
        "ranker_train_rows": int(len(df)),
        "ranker_positive_rate": float(df.label.mean()),
        "ranker_auc": [float(x) for x in hist.history.get("auc", [])],
        "ranker_loss": [float(x) for x in hist.history["loss"]],
        "ranker_seconds": time.time() - t0,
    }


def run_all(cfg: Config) -> dict:
    cfg.save()
    metrics = {**train_retrieval(cfg), **train_ranker(cfg)}
    (cfg.art / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
