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


class TopicConfig(BaseModel):
    name: str
    paths: List[str]
    target_ratio: Optional[float] = None
    repetition_factor: float = 1.0
    max_topic_ratio: float = 1.0

class SourceConfig(BaseModel):
    name: str
    paths: list[str]
    topics: Optional[list[TopicConfig]] = None
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
    entity: str = "ai2-llm"
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


class AnnealConfig(BaseModel):
    enabled: bool = True
    initial_lr: Optional[float] = None


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
    annealing: Optional[AnnealConfig] = None
    nccl_debug: bool = False
    activation_checkpointing: bool = False
    model_overrides: Optional[List[str]] = None
    scheduler_type: SchedulerType = SchedulerType.COS_LINEAR
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
        if value is not None and info.data.get("load_path") is None:
            raise ValueError("If annealing is enabled, load_path must be specified.")
        return value


class SwarmConfig(ExperimentConfig):
    """Additional configuration for launching a mixing swarm."""
    variants: int = 1  # number of variants to generate
    allow_repetition: bool = False  # whether to allow repetition of sources in mixtures
    minimum_weight: float = 0.05  # minimum weight for sources in mixtures
    fixed_source_weights: Optional[dict[str, float]] = None  # fixed weights for sources in mixtures
    sample_multiplier: int = 10  # multiplier for the number of samples per source
    min_source_strength: Optional[float] = None  # minimum strength for sources
    max_source_strength: Optional[float] = None  # maximum strength for sources
    min_topic_strength: Optional[float] = None  # minimum strength for topics
    max_topic_strength: Optional[float] = None  # maximum strength for topics
    min_strength: float = 0.1  # minimum strength for sources and topics
    max_strength: float = 5.0  # maximum strength for sources and topics


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
