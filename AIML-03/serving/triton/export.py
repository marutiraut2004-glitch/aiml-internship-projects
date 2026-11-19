"""Copy the ranker SavedModel into a Triton model repository and write its config.pbtxt."""
from __future__ import annotations

import shutil
from pathlib import Path

from recsys.config import Config
from recsys.models import RANKER_INPUTS

REPO = Path(__file__).parent / "model_repository"


def config_pbtxt() -> str:
    dtype = {"int32": "TYPE_INT32", "float32": "TYPE_FP32"}
    inputs = ",\n".join(
        f'  {{ name: "{n}", data_type: {dtype[dt.name]}, dims: [ -1 ] }}' for n, dt in RANKER_INPUTS.items()
    )
    return f'''name: "ranker"
platform: "tensorflow_savedmodel"
max_batch_size: 0
input [
{inputs}
]
output [
  {{ name: "score", data_type: TYPE_FP32, dims: [ -1 ] }}
]
instance_group [ {{ count: 1, kind: KIND_CPU }} ]
'''


def export(cfg: Config) -> Path:
    dst = REPO / "ranker"
    target = dst / "1" / "model.savedmodel"
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cfg.art / "ranker_savedmodel", target)
    (dst / "config.pbtxt").write_text(config_pbtxt())
    return dst
