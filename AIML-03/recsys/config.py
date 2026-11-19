from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # data scale
    n_users: int = 5000
    n_items: int = 2000
    n_categories: int = 20
    n_age_buckets: int = 6
    n_countries: int = 8
    n_price_bins: int = 10
    interactions_per_user: int = 40
    holdout_per_user: int = 5
    seed: int = 7

    # models
    embedding_dim: int = 32
    retrieval_epochs: int = 8
    ranking_epochs: int = 8
    retrieval_batch: int = 2048
    ranking_batch: int = 1024
    neg_per_pos: int = 3

    # serving
    n_candidates: int = 200
    top_k: int = 10
    index_kind: str = "hnsw"  # flat | hnsw | ivf
    rec_cache_ttl: int = 60

    data_dir: str = field(default_factory=lambda: os.environ.get("REC_DATA_DIR", str(ROOT / "data")))
    artifact_dir: str = field(default_factory=lambda: os.environ.get("REC_ARTIFACT_DIR", str(ROOT / "artifacts")))

    @property
    def raw_dir(self) -> Path:
        return Path(self.data_dir) / "raw"

    @property
    def processed_dir(self) -> Path:
        return Path(self.data_dir) / "processed"

    @property
    def art(self) -> Path:
        return Path(self.artifact_dir)

    def save(self) -> None:
        self.art.mkdir(parents=True, exist_ok=True)
        (self.art / "config.json").write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, artifact_dir: str | None = None) -> "Config":
        d = Path(artifact_dir or os.environ.get("REC_ARTIFACT_DIR", ROOT / "artifacts"))
        return cls(**json.loads((d / "config.json").read_text()))
