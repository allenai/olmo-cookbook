from dataclasses import dataclass
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, List, Optional, Union

from olmo_core.optim.scheduler import CosWithWarmup, CosWithWarmupAndLinearDecay, LinearWithWarmup, WSD, Scheduler
from olmo_core.optim import SchedulerUnits
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


class WrappedScheduler:
    COSINE = CosWithWarmup
    COSINE_LINEAR = CosWithWarmupAndLinearDecay
    LINEAR = LinearWithWarmup
    WSD = WSD

    @classmethod
    def from_name_and_config(cls, name: str, config: dict[str, Any]) -> Scheduler:
        """Get the scheduler class by name and config"""

        # TODO(undfined): Remove this temporary fix for issue in Scheduler conditional check
        # where decay_fraction = None is required
        if "decay" in config:
            config["decay_fraction"] = None

        return getattr(cls, name)(**config)


class SchedulerConfig(BaseModel):
    scheduler: str = "COSINE_LINEAR"
    units: SchedulerUnits = SchedulerUnits.steps
    warmup: Optional[int] = None
    warmup_fraction: Optional[float] = None
    decay: Optional[int] = None
    decay_fraction: Optional[float] = None

    @classmethod
    def _validate_mutually_exclusive(cls, v, field_name, other_field_name, info):
        """Helper method to validate that two fields are mutually exclusive."""
        if v is not None and info.data.get(other_field_name) is not None:
            raise ValueError(
                f"{field_name} and {other_field_name} are mutually exclusive and cannot both be specified."
            )
        return v

    @field_validator("warmup")
    @classmethod
    def validate_warmup(cls, v, info):
        """Validate that warmup and warmup_fraction are not both specified."""
        return cls._validate_mutually_exclusive(v, "warmup", "warmup_fraction", info)

    @field_validator("warmup_fraction")
    @classmethod
    def validate_warmup_fraction(cls, v, info):
        """Validate that warmup and warmup_fraction are not both specified."""
        return cls._validate_mutually_exclusive(v, "warmup_fraction", "warmup", info)

    @field_validator("decay")
    @classmethod
    def validate_decay(cls, v, info):
        """Validate that decay is mutually exclusive with decay_fraction"""
        return cls._validate_mutually_exclusive(v, "decay", "decay_fraction", info)

    @field_validator("decay_fraction")
    @classmethod
    def validate_decay_fraction(cls, v, info):
        """Validate that decay_fraction is mutually exclusive with decay"""
        return cls._validate_mutually_exclusive(v, "decay_fraction", "decay", info)


class AnnealConfig(BaseModel):
    enabled: bool = True
    initial_lr: Optional[float] = None


class BatchSizeWarmupConfig(BaseModel):
    # Batch size multipliers
    # e.g. [1, 2, 4] means 1x, 2x, and 4x the base global_batch_size
    batches: list[int]
    # Floats representing transition points in the schedule
    # e.g. [0.0, 0.5, 1.0] means at 0% of the training, use 1x batch size, at 50% of the training, use 2x batch size, and at 100% of the training, use 4x batch size
    schedule: list[float]


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
    scheduler_config: SchedulerConfig
    batch_size_warmup: Optional[BatchSizeWarmupConfig] = None
    load_path: Optional[str] = None
    load_state: bool = True
    annealing: Optional[AnnealConfig] = None
    nccl_debug: bool = False
    activation_checkpointing: bool = False
    weight_decay: Optional[float] = None
    model_overrides: Optional[List[str]] = None
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
