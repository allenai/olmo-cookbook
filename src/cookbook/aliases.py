from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, List, Optional, Union

from olmo_core.data.types import NumpyDatasetDType
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.train.common import Duration
from pydantic import BaseModel, field_validator

from cookbook.model.config import ModelConfigIdentifier
from cookbook.model.evaluators import DownstreamEvaluator

DownstreamEvaluatorType = Union[str, DownstreamEvaluator]
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


class MetricBackend(Enum):
    wandb = "wandb"
    comet = "comet"


class MetricsConfig(BaseModel):
    project: str = "olmo-cookbook"
    workspace: str = "ai2"
    backends: list[MetricBackend] = [MetricBackend.wandb]


class SchedulerType(Enum):
    COSINE = "cosine"
    COS_LINEAR = "cos_linear"
    LINEAR = "linear"
    WSD = "wsd"

    @classmethod
    def values(cls):
        return [e.value for e in cls]

    @classmethod
    def keys(cls):
        return [e.name for e in cls]


class ExperimentConfig(BaseModel, extra="forbid"):
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
    model: ModelConfigIdentifier
    load_path: Optional[str] = None
    load_state: bool = True
    annealing: bool = False
    activation_checkpointing: bool = False
    model_overrides: Optional[List[str]] = None
    scheduler_type: SchedulerType = SchedulerType.COSINE
    hard_stop: Optional[Duration] = None
    rank_microbatch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    global_batch_size: Optional[int] = None
    lm_evaluator: bool = False
    downstream_evaluators: list[DownstreamEvaluatorType] = []  # type: ignore
    max_target_sequence_length: int = 8192
    metrics_config: Optional[MetricsConfig] = MetricsConfig()
    preemptible: bool = True
    shared_filesystem: bool = False
    weka: bool = False
    eval_interval: int = 200
    save_interval: int = 1000
    warmup_steps: Optional[int] = None
    path: Path

    @field_validator("model", mode="before")
    @classmethod
    def validate_model(cls, value):
        """Convert string to ModelConfigIdentifier if needed."""
        if isinstance(value, str):
            return ModelConfigIdentifier(value)
        return value

    @field_validator("annealing")
    @classmethod
    def validate_annealing(cls, value, info):
        """Validate that if annealing is True, then load_path must not be None."""
        if value is True and info.data.get("load_path") is None:
            raise ValueError("If annealing is enabled, load_path must be specified.")
        return value


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
