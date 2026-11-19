"""Synthetic data with real latent structure, so the models have something to learn.

Users and items live in a shared latent space. Items belong to categories, users have
age bucket and country effects, and every item has a popularity bias. Interactions are
sampled without replacement from softmax(affinity) using the Gumbel top-k trick.

The loader for MovieLens in `movielens.py` writes the same three CSV files, so every
later stage works on either source.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

LATENT = 8


def generate(cfg: Config) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)

    cat_dir = rng.normal(size=(cfg.n_categories, LATENT))
    item_cat = rng.integers(0, cfg.n_categories, cfg.n_items)
    V = rng.normal(size=(cfg.n_items, LATENT)) * 0.7 + cat_dir[item_cat]
    price = np.round(rng.lognormal(3.0, 0.8, cfg.n_items), 2)
    price_bin = pd.qcut(price, cfg.n_price_bins, labels=False, duplicates="drop").astype(int)
    pop_bias = rng.normal(0, 0.6, cfg.n_items)

    age = rng.integers(0, cfg.n_age_buckets, cfg.n_users)
    country = rng.integers(0, cfg.n_countries, cfg.n_users)
    age_dir = rng.normal(size=(cfg.n_age_buckets, LATENT))
    country_dir = rng.normal(size=(cfg.n_countries, LATENT))
    U = rng.normal(size=(cfg.n_users, LATENT)) * 0.7 + 0.6 * age_dir[age] + 0.6 * country_dir[country]

    logits = 2.0 * (U @ V.T) / np.sqrt(LATENT) + pop_bias
    noisy = logits + rng.gumbel(size=logits.shape)
    n = min(cfg.interactions_per_user, cfg.n_items - 1)
    top = np.argpartition(-noisy, n, axis=1)[:, :n]

    users = np.repeat(np.arange(cfg.n_users), n)
    items = top.reshape(-1)
    ts = rng.integers(0, 90 * 24 * 3600, users.size)

    return {
        "users": pd.DataFrame({"user_id": np.arange(cfg.n_users), "age_bucket": age, "country": country}),
        "items": pd.DataFrame(
            {"item_id": np.arange(cfg.n_items), "category": item_cat, "price": price, "price_bin": price_bin}
        ),
        "interactions": pd.DataFrame({"user_id": users, "item_id": items, "ts": ts}),
    }


def write_raw(cfg: Config) -> None:
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    for name, df in generate(cfg).items():
        df.to_csv(cfg.raw_dir / f"{name}.csv", index=False)
