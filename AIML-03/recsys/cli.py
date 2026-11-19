"""python -m recsys.cli <command>   (see README for the full run order)"""
from __future__ import annotations

import argparse
import json
import logging

from .config import Config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="recsys")
    ap.add_argument("command", choices=[
        "generate", "movielens", "etl", "train-retrieval", "train-ranker", "train",
        "evaluate", "export-triton", "all"])
    ap.add_argument("--users", type=int)
    ap.add_argument("--items", type=int)
    ap.add_argument("--path", help="ml-100k folder for the movielens command")
    ap.add_argument("--index", choices=["flat", "hnsw", "ivf"])
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    cfg = Config.load() if a.command in {"train-ranker", "evaluate", "export-triton"} else Config()
    if a.users:
        cfg.n_users = a.users
    if a.items:
        cfg.n_items = a.items
    if a.index:
        cfg.index_kind = a.index

    def gen():
        from .data_gen import write_raw
        write_raw(cfg)

    def etl():
        from .spark_etl import run
        run(cfg)

    def ev():
        from .evaluate import evaluate
        print(json.dumps(evaluate(cfg), indent=2))

    def tr(fn):
        from . import train
        cfg.save()
        print(json.dumps(getattr(train, fn)(cfg), indent=2))

    def exp():
        from serving.triton.export import export
        export(cfg)

    cmds = {
        "generate": gen, "etl": etl,
        "train-retrieval": lambda: tr("train_retrieval"),
        "train-ranker": lambda: tr("train_ranker"),
        "train": lambda: tr("run_all"),
        "evaluate": ev, "export-triton": exp,
    }
    if a.command == "movielens":
        from .movielens import load
        cfg = load(a.path, cfg)
        cfg.save()
    elif a.command == "all":
        for step in (gen, etl, lambda: tr("run_all"), ev):
            step()
    else:
        cmds[a.command]()


if __name__ == "__main__":
    main()
