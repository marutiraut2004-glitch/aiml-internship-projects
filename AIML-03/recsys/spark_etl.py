"""PySpark ETL: raw CSVs -> train/test split and feature tables (parquet).

Per user, by time: the most recent `holdout_per_user` interactions are the test set, the
`holdout_per_user` before that are a validation slice, and the rest is training data.

The retrieval model trains on `train` only. The ranker trains on `val`, which the retrieval
model has never seen, so the retrieval scores it learns from look like the scores it will
see at serving time. Popularity and activity features use train+val (all known history),
never the test set.
"""
from __future__ import annotations

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from .config import Config


def run(cfg: Config) -> None:
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("recsys-etl")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    raw, out = str(cfg.raw_dir), cfg.processed_dir
    out.mkdir(parents=True, exist_ok=True)

    read = lambda name: spark.read.csv(f"{raw}/{name}.csv", header=True, inferSchema=True)
    users, items, inter = read("users"), read("items"), read("interactions")

    inter = inter.dropDuplicates(["user_id", "item_id"])
    w = Window.partitionBy("user_id").orderBy(F.col("ts").desc(), F.col("item_id"))
    ranked = inter.withColumn("rn", F.row_number().over(w))
    h = cfg.holdout_per_user
    test = ranked.filter(F.col("rn") <= h).drop("rn")
    val = ranked.filter((F.col("rn") > h) & (F.col("rn") <= 2 * h)).drop("rn")
    train = ranked.filter(F.col("rn") > 2 * h).drop("rn")
    history = train.unionByName(val)

    item_pop = history.groupBy("item_id").agg(F.count("*").alias("item_pop"))
    item_feat = (
        items.join(item_pop, "item_id", "left")
        .fillna({"item_pop": 0})
        .withColumn("item_pop_log", F.log1p(F.col("item_pop")))
        .select("item_id", "category", "price_bin", "item_pop", "item_pop_log")
    )
    user_act = history.groupBy("user_id").agg(F.count("*").alias("user_activity"))
    user_feat = (
        users.join(user_act, "user_id", "left")
        .fillna({"user_activity": 0})
        .withColumn("user_activity_log", F.log1p(F.col("user_activity")))
        .select("user_id", "age_bucket", "country", "user_activity", "user_activity_log")
    )

    for name, df in [("train", train), ("val", val), ("test", test), ("item_features", item_feat), ("user_features", user_feat)]:
        df.orderBy(df.columns[0]).coalesce(1).write.mode("overwrite").parquet(str(out / name))
    spark.stop()
