from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass, fields as dataclass_fields, field as dataclass_field
from typing import List, ClassVar, Self, TypeVar, Generic

import requests
from tqdm import tqdm


BASE_URL = "https://oe-eval-datalake.allen.ai"
FROM_CREATED_DT = "2024-07-01"


T = TypeVar("T")


@dataclass
class BaseDatalakeItem(Generic[T]):
    _base_url: ClassVar[str] = "https://oe-eval-datalake.allen.ai"
    _from_created_dt: ClassVar[str] = "2024-07-01"

    def __post_init__(self):
        for field in dataclass_fields(self):
            # we check if a field has a parser function and use to initialize it
            if (parser := field.metadata.get("parser", None)) is not None:
                setattr(self, field.name, parser(getattr(self, field.name)))

    @classmethod
    def run(cls, **kwargs: T) -> List[Self]:
        """Run a query to retrieve one or more datalake items of this type."""
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def prun(cls, num_workers: int | None = None, **kwargs: list[T]) -> list[Self]:
        """Same as run() method, but runs in parallel.

        Args:
            num_workers: Number of workers to use. If None, the number of workers
                will be the default in ThreadPoolExecutor.
            **kwargs: Keyword arguments to pass to the run() method.

        Returns:
            List of datalake items.
        """
        num_args = -1
        for k, v in kwargs.items():
            assert isinstance(v, list), f"Argument {k} must be a list"
            num_args = len(v) if num_args == -1 else num_args
            assert len(v) == num_args, f"Argument {k} must have the same length as the previous arguments"
        assert num_args > 0, "At least one argument must be provided"

        results: list[Self] = []
        with ExitStack() as stack:
            pool = stack.enter_context(ThreadPoolExecutor(max_workers=num_workers))
            pbar = stack.enter_context(tqdm(total=num_args, desc=cls.__name__))

            futures = [pool.submit(cls.run, **{k: v[i] for k, v in kwargs.items()}) for i in range(num_args)]
            for future in as_completed(futures):
                try:
                    results.extend(future.result())
                except Exception as e:
                    for remaining_future in futures:
                        remaining_future.cancel()
                    raise e
                pbar.update(1)

        return results

@dataclass
class Tag:
    """
    A tag is a key-value pair; it is returned as a string like "key=value,key2=value2".
    """
    key: str
    value: str

    def __post_init__(self):
        assert "=" not in self.key and "=" not in self.value, "Key and value cannot contain '=' character"
        assert "," not in self.key and "," not in self.value, "Key and value cannot contain ',' character"

    def __str__(self) -> str:
        return f"{self.key}={self.value}"

    @classmethod
    def from_str(cls, s: str) -> list[Self]:
        return [cls(*pair.split("=", 1)) for pair in s.split(",")] if (s := s.strip()) else []


@dataclass
class FindExperiments(BaseDatalakeItem):
    """Find all experiments for a given dashboard."""

    experiment_id: str
    model_name: str
    # parser argument will make sure that nested dataclasses are initialized
    tags: list[Tag] = dataclass_field(default_factory=list, metadata=dict(parser=Tag.from_str))

    _endpoint: ClassVar[str] = "bluelake/find-experiments/"

    @classmethod
    def run(cls, dashboard: str, limit: int = 10_000) -> list[Self]:
        response = requests.get(
            f"{cls._base_url}/{cls._endpoint}",
            params={
                "from_created_dt": cls._from_created_dt,
                "tags": f"dashboard={dashboard}",
                "limit": limit,
                "return_fields": ",".join(f.name for f in dataclass_fields(cls)),
            },
            headers={"accept": "application/json"},
        )
        response.raise_for_status()
        return [cls(**experiment) for experiment in response.json()]


@dataclass
class Metrics:
    """
    A collection of values for a given task and model.
    """
    primary_score: float
    logits_per_byte_corr: float | None = None
    bits_per_byte_corr: float | None = None

    # extra metrics are task-specific and are sometimes returned as a dictionary
    extra_metrics: dict = dataclass_field(default_factory=dict)

    @property
    def bpb(self) -> float | None:
        return self.bits_per_byte_corr or self.logits_per_byte_corr

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        # these are the names of the fields that are shared across tasks
        fields = {f.name for f in dataclass_fields(cls)}

        # move all metrics that are not shared across tasks to extra_metrics
        extra_metrics = {**d.pop("extra_metrics", {}), **{k: d.pop(k) for k in list(d) if k not in fields}}
        return cls(**d, extra_metrics=extra_metrics)


@dataclass
class MetricsAll(BaseDatalakeItem):
    """Find all metrics for a given experiment."""

    compute_config: dict
    current_date: str
    metrics: Metrics = dataclass_field(metadata=dict(parser=Metrics.from_dict))
    model_config: dict
    model_hash: str
    num_instances: int
    processing_time: float
    task_config: dict
    task_hash: str
    task_idx: int
    task_name: str

    # this is not a field in dataclass
    _endpoint: ClassVar[str] = "greenlake/metrics-all/"

    @property
    def alias(self) -> str | None:
        """
        The alias used to identify the task when launching experiments.
        """
        return self.task_config.get("metadata", {}).get("alias", None)

    @property
    def model_path(self) -> str | None:
        return self.model_config.get("model_path", None)

    @property
    def model_name(self) -> str | None:
        return self.model_config.get("model", None)

    @property
    def is_aggregate(self) -> bool:
        return self.task_config.get("metadata", {}).get("num_tasks", 0) > 0

    @classmethod
    def run(cls, experiment_id: str) -> List[Self]:
        response = requests.get(
            f"{cls._base_url}/{cls._endpoint.rstrip('/')}/{experiment_id}",
            headers={"accept": "application/json"},
        )
        response.raise_for_status()
        return [cls(**metric) for metric in response.json()]
