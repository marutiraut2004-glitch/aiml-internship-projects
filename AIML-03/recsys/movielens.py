"""Optional: build the same raw CSVs from the public MovieLens 100K dataset.

Download ml-100k from https://grouplens.org/datasets/movielens/100k/ and unzip it, then:

    python -m recsys.cli movielens --path /path/to/ml-100k

Ratings >= 4 are treated as positive interactions. Users get age and (hashed) zip-region
buckets, items use their first listed genre as the category. There is no price in
MovieLens, so release year is binned instead and stored in the same `price_bin` column.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

GENRES = ["unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
          "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
          "Thriller", "War", "Western"]


def load(path: str, cfg: Config) -> Config:
    p = Path(path)
    r = pd.read_csv(p / "u.data", sep="\t", names=["user_id", "item_id", "rating", "ts"])
    r = r[r.rating >= 4]
    u = pd.read_csv(p / "u.user", sep="|", names=["user_id", "age", "gender", "occupation", "zip"])
    i = pd.read_csv(p / "u.item", sep="|", encoding="latin-1", header=None)

    # re-index ids to 0..n-1
    uid = {v: k for k, v in enumerate(sorted(u.user_id.unique()))}
    iid = {v: k for k, v in enumerate(sorted(i[0].unique()))}
    r = r[r.user_id.isin(uid) & r.item_id.isin(iid)].copy()
    r["user_id"] = r.user_id.map(uid)
    r["item_id"] = r.item_id.map(iid)

    n_age = 6
    users = pd.DataFrame({
        "user_id": u.user_id.map(uid),
        "age_bucket": pd.cut(u.age, [0, 18, 25, 35, 45, 55, 120], labels=False),
        "country": u.zip.astype(str).str[:1].map(lambda s: ord(s[0]) % 8),
    }).sort_values("user_id")

    genre_cols = i.iloc[:, 5:24].to_numpy()
    category = genre_cols.argmax(axis=1)
    year = pd.to_datetime(i[2], errors="coerce").dt.year.fillna(1995)
    items = pd.DataFrame({
        "item_id": i[0].map(iid),
        "category": category,
        "price": year,
        "price_bin": pd.qcut(year.rank(method="first"), 10, labels=False).astype(int),
    }).sort_values("item_id")

    cfg.n_users, cfg.n_items = len(users), len(items)
    cfg.n_categories, cfg.n_age_buckets, cfg.n_countries, cfg.n_price_bins = len(GENRES), n_age, 8, 10
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    users.to_csv(cfg.raw_dir / "users.csv", index=False)
    items.to_csv(cfg.raw_dir / "items.csv", index=False)
    r[["user_id", "item_id", "ts"]].to_csv(cfg.raw_dir / "interactions.csv", index=False)
    return cfg
