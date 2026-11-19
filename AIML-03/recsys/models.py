"""TensorFlow Recommenders models: two-tower retrieval and a feature-rich ranker."""
from __future__ import annotations

import tensorflow as tf
import tensorflow_recommenders as tfrs

from .config import Config

K = tf.keras


class UserTower(K.Model):
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.embedding_dim
        self.user = K.layers.Embedding(cfg.n_users, d)
        self.age = K.layers.Embedding(cfg.n_age_buckets, 8)
        self.country = K.layers.Embedding(cfg.n_countries, 8)
        self.mlp = K.Sequential([K.layers.Dense(64, activation="relu"), K.layers.Dense(d)])

    def call(self, f):
        x = tf.concat([self.user(f["user_id"]), self.age(f["age_bucket"]), self.country(f["country"])], -1)
        return tf.math.l2_normalize(self.mlp(x), axis=-1)


class ItemTower(K.Model):
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.embedding_dim
        self.item = K.layers.Embedding(cfg.n_items, d)
        self.cat = K.layers.Embedding(cfg.n_categories, 8)
        self.price = K.layers.Embedding(cfg.n_price_bins, 4)
        self.mlp = K.Sequential([K.layers.Dense(64, activation="relu"), K.layers.Dense(d)])

    def call(self, f):
        x = tf.concat([self.item(f["item_id"]), self.cat(f["category"]), self.price(f["price_bin"])], -1)
        return tf.math.l2_normalize(self.mlp(x), axis=-1)


class TwoTower(tfrs.Model):
    """Stage 1. In-batch softmax with logQ correction (popularity debiasing) and
    removal of accidental hits (same item appearing twice in a batch)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.user_tower = UserTower(cfg)
        self.item_tower = ItemTower(cfg)
        self.task = tfrs.tasks.Retrieval(temperature=0.1)

    def compute_loss(self, features, training=False):
        u = self.user_tower(features)
        v = self.item_tower(features)
        return self.task(
            u, v,
            candidate_sampling_probability=features["item_prob"],
            candidate_ids=features["item_id"],
            compute_metrics=False,
        )


class RankerNet(K.Model):
    """Scores one (user, item) pair from features only: user side features, item side
    features, popularity/activity, and the stage 1 retrieval score.

    It deliberately has no user_id / item_id embeddings. The ranker is trained on only a few
    positives per user, and ID embeddings memorise them (train AUC ~0.999, worse test ranking).
    The retrieval model already carries the ID signal through `retr_score`."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.age = K.layers.Embedding(cfg.n_age_buckets, 8)
        self.country = K.layers.Embedding(cfg.n_countries, 8)
        self.cat = K.layers.Embedding(cfg.n_categories, 8)
        self.price = K.layers.Embedding(cfg.n_price_bins, 4)
        self.mlp = K.Sequential([
            K.layers.Dense(64, activation="relu"),
            K.layers.Dropout(0.2),
            K.layers.Dense(32, activation="relu"),
            K.layers.Dense(1),
        ])
        # Skip connection: the ranker starts from the retrieval ordering and learns a correction.
        self.retr_weight = self.add_weight(name="retr_weight", shape=(), initializer="ones")

    def call(self, f, training=False):
        dense = tf.stack([f["item_pop_log"], f["user_activity_log"], f["retr_score"] * 10.0], -1)
        x = tf.concat([self.age(f["age_bucket"]), self.country(f["country"]),
                       self.cat(f["category"]), self.price(f["price_bin"]), dense], -1)
        return tf.squeeze(self.mlp(x, training=training), -1) + self.retr_weight * f["retr_score"] * 10.0


class RankingModel(tfrs.Model):
    def __init__(self, cfg: Config):
        super().__init__()
        self.net = RankerNet(cfg)
        self.task = tfrs.tasks.Ranking(
            loss=K.losses.BinaryCrossentropy(from_logits=True),
            metrics=[K.metrics.AUC(from_logits=True, name="auc")],
        )

    def compute_loss(self, features, training=False):
        return self.task(labels=features["label"], predictions=self.net(features, training=training))


# Feature names and dtypes the ranker expects. Triton and the local SavedModel use the same list.
RANKER_INPUTS = {
    "age_bucket": tf.int32, "country": tf.int32,
    "category": tf.int32, "price_bin": tf.int32,
    "item_pop_log": tf.float32, "user_activity_log": tf.float32, "retr_score": tf.float32,
}


class RankerServing(tf.Module):
    def __init__(self, net: RankerNet):
        super().__init__()
        self.net = net

    @tf.function(input_signature=[tf.TensorSpec([None], dt, name=n) for n, dt in RANKER_INPUTS.items()])
    def serve(self, age_bucket, country, category, price_bin, item_pop_log, user_activity_log, retr_score):
        f = dict(age_bucket=age_bucket, country=country,
                 category=category, price_bin=price_bin, item_pop_log=item_pop_log,
                 user_activity_log=user_activity_log, retr_score=retr_score)
        return {"score": tf.sigmoid(self.net(f))}
