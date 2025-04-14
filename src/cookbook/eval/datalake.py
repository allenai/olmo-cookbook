from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass, fields as dataclass_fields, field as dataclass_field
from functools import partial
from typing import List, ClassVar, Self, TypeVar, Generic, Callable
from threading import main_thread, current_thread
import hashlib
import json

import requests
from tqdm import tqdm

T = TypeVar("T")
V = TypeVar("V")


@dataclass
class BaseDatalakeItem(Generic[T]):
    _base_url: ClassVar[str] = "https://oe-eval-datalake.allen.ai"
    _from_created_dt: ClassVar[str] = "2024-07-01"

    def __post_init__(self):
        for field in dataclass_fields(self):
            # we check if a field has a parser function and use to initialize it
            if (parser := field.metadata.get("parser", None)) is not None:
                setattr(self, field.name, parser(getattr(self, field.name)))

    @staticmethod
    def _is_prun() -> bool:
        return current_thread() != main_thread()

    @classmethod
    def run(cls, **kwargs: T) -> List[Self]:
        """Run a query to retrieve one or more datalake items of this type."""
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def _prun(
        cls,
        fns: list[Callable[[], V]],
        num_workers: int | None = None,
        quiet: bool = False
    ) -> list[V]:
        # Validate input arguments

        results: list[V] = []
        with ExitStack() as stack:
            # Set up thread pool and progress bar
            pool = stack.enter_context(ThreadPoolExecutor(max_workers=num_workers))
            pbar = stack.enter_context(tqdm(total=len(fns), desc=cls.__name__, disable=quiet))

            # Submit all tasks to the thread pool
            futures = [pool.submit(fn) for fn in fns]

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    # Cancel all remaining tasks if any task fails
                    for remaining_future in futures:
                        remaining_future.cancel()
                    raise e
                pbar.update(1)

        return results

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

        fns = [partial(cls.run, **{k: v[i] for k, v in kwargs.items()}) for i in range(num_args)]

        # actually run the function in parallel
        results = cls._prun(fns=fns, num_workers=num_workers, quiet=False)
        return [result for result_group in results for result in result_group]



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
    def run(
        cls,
        dashboard: str | None = None,
        model_name: str | None = None,
        limit: int = 10_000
    ) -> list[Self]:

        # make sure at least one of dashboard or model_name is provided
        assert dashboard or model_name, "Either dashboard or model_name must be provided"
        response = requests.get(
            f"{cls._base_url}/{cls._endpoint}",
            params={
                "from_created_dt": cls._from_created_dt,
                **({"tags": f"dashboard={dashboard}"} if dashboard else {}),
                **({"model_name": model_name} if model_name else {}),
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
    def alias(self) -> str:
        """
        The alias used to identify the task when launching experiments.
        """
        task_alias = self.task_config.get("metadata", {}).get("alias", None)

        if task_alias is None:
            # we hash the task_config in json format and get the first 6 characters as suffix
            task_suffix = hashlib.sha256(json.dumps(self.task_config).encode()).hexdigest()[:6]
            task_alias = f"{self.task_name}-{task_suffix}"

        return task_alias

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


@dataclass
class RemoveFromDashboard(BaseDatalakeItem):
    _endpoint: ClassVar[str] = "bluelake/add-experiment-tags/"

    @classmethod
    def _endpoint_request(cls, experiment_id: str, tags: list[Tag], overwrite: bool = True) -> Self:
        params = {
            "tags": ",".join([str(tag) for tag in tags]),
            "overwrite": overwrite,
            "experiment_id": experiment_id,
        }
        response = requests.put(
            f"{cls._base_url}/{cls._endpoint}",
            headers={"accept": "application/json"},
            params=params,
        )
        response.raise_for_status()
        return cls(**(response.json() or {}))

    @classmethod
    def run(cls, model_name: str, dashboard: str) -> List[Self]:
        runs = FindExperiments.run(model_name=model_name, dashboard=dashboard)

        fns = []
        for run in runs:
            new_tags = [tag for tag in run.tags if (tag.key != "dashboard" or tag.value != dashboard)]
            fn = partial(cls._endpoint_request, experiment_id=run.experiment_id, tags=new_tags, overwrite=True)
            fns.append(fn)

        if cls._is_prun():
            # if we are already running in parallel, then we can use _prun() to create more parallelism
            # (we avoid making a second progress bar by setting quiet=True)
            return cls._prun(fns=fns, num_workers=len(runs), quiet=True)
        else:
            # if we are single-threaded, then we can just run the function sequentially
            return [fn() for fn in fns]


@dataclass
class AddToDashboard(RemoveFromDashboard):
    @classmethod
    def run(cls, model_name: str, dashboard: str) -> List[Self]:
        runs = FindExperiments.run(model_name=model_name)

        fns = []
        for run in runs:
            # we first check if the dashboard tag is already present
            if any(tag.key == "dashboard" and tag.value == dashboard for tag in run.tags):
                continue

            # if it is not present, then we add it
            new_tags = [Tag(key="dashboard", value=dashboard)]
            fn = partial(cls._endpoint_request, experiment_id=run.experiment_id, tags=new_tags, overwrite=False)
            fns.append(fn)

        if cls._is_prun():
            # if we are already running in parallel, then we can use _prun() to create more parallelism
            # (we avoid making a second progress bar by setting quiet=True)
            return cls._prun(fns=fns, num_workers=len(runs), quiet=True)
        else:
            # if we are single-threaded, then we can just run the function sequentially
            return [fn() for fn in fns]
