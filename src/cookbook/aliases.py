from os import PathLike
from pathlib import Path
from typing import Any, Optional, Union

from olmo_core.data.types import NumpyDatasetDType
from olmo_core.launch.beaker import BeakerLaunchConfig
from pydantic import BaseModel

from cookbook.model.evaluators import DownstreamEvaluator

PathType = Union[Path, PathLike[Any], str]

try:
    from beaker import Priority  # pyright: ignore
except ImportError:
    Priority = str


class SourceConfig(BaseModel):
    name: str
    paths: list[str]
    target_ratio: Optional[float] = None
    repetition_factor: float = 1.0
    max_source_ratio: float = 1.0


class SourceInstance(BaseModel):
    name: str
    paths: list[str]
    ratio: float
    repetition_factor: float = 1.0


class DatasetConfig(BaseModel):
    sources: list[SourceConfig]
    dtype: NumpyDatasetDType = NumpyDatasetDType.uint32
    processes: int = 16
    seed: int = 42


class WandbConfig(BaseModel):
    project: str


class ExperimentConfig(BaseModel):
    name: str
    description: str
    budget: str
    workspace: str
    nodes: int
    gpus: int
    max_tokens: int
    sequence_length: int
    seed: int
    cluster: str
    tokenizer: str
    priority: Priority  # pyright: ignore
    dataset: DatasetConfig
    tokenizer: str
    model: str
    load_path: Optional[str] = None
    rank_microbatch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    global_batch_size: Optional[int] = None
    lm_evaluator: bool = False
    downstream_evaluators: list[DownstreamEvaluator] = []
    max_target_sequence_length: int = 8192
    wandb: Optional[WandbConfig] = None
    preemptible: bool = True
    shared_filesystem: bool = False
    weka: bool = False
    eval_interval: int = 200
    save_interval: int = 1000
    path: Path


class ExperimentInstance(BaseModel):
    name: str
    sources: list[SourceInstance]


class ExperimentGroup(BaseModel):
    config: ExperimentConfig
    group_id: str
    instances: list[ExperimentInstance]


class LaunchGroup(BaseModel):
    instances: list[BeakerLaunchConfig]


def validate_sources(sources: list[SourceConfig]):
    """Validate a list of source configurations."""
    target_ratio_present = any(source.target_ratio is not None for source in sources)

    for source in sources:
        if target_ratio_present and source.target_ratio is None:
            raise ValueError("If any source has target_ratio set, all sources must have target_ratio set.")
